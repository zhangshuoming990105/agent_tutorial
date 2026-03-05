#!/usr/bin/env python3
"""Simplified compile script: force-compile root and kernels CUDA/C++ sources."""

import os
import re
import shutil
import sys
import traceback
from pathlib import Path

import torch
import torch.utils.cpp_extension as cpp_ext


def detect_backend() -> str:
    """Detect whether this PyTorch build targets CUDA or HIP."""
    if getattr(torch.version, 'hip', None):
        return 'hip'
    if torch.version.cuda is not None:
        return 'cuda'
    return 'none'


def _has_hip_counterpart(name: str, all_names: set[str]) -> str | None:
    """Return the hipified filename if it already exists on disk, else None.

    Handles common hipify naming conventions:
      foo.cpp         → foo_hip.cpp        (append _hip)
      foo_cuda.cpp    → foo_cuda_hip.cpp   (append _hip)
      foo_cuda_bar.cpp→ foo_hip_bar.cpp    (replace cuda→hip)
    """
    stem = Path(name).stem
    # Pattern 1: foo.cpp → foo_hip.cpp
    cand1 = stem + '_hip.cpp'
    if cand1 in all_names:
        return cand1
    # Pattern 2: replace 'cuda' with 'hip' in the filename
    if 'cuda' in name:
        cand2 = name.replace('cuda', 'hip')
        if cand2 in all_names and cand2 != name:
            return cand2
    return None


def find_sources(backend: str) -> list[str]:
    root = Path('.')
    kernels_dir = Path('kernels')

    root_sources = [str(p) for p in root.glob('*.cu')] + [str(p) for p in root.glob('*.cpp')]
    kernel_sources: list[str] = []
    if kernels_dir.is_dir():
        kernel_cu = list(kernels_dir.glob('*.cu'))
        kernel_hip = list(kernels_dir.glob('*.hip'))
        kernel_cpp = list(kernels_dir.glob('*.cpp'))

        if backend == 'hip':
            # --- .cu / .hip files ---
            # Prefer .hip when the hipified version already exists on disk;
            # otherwise keep .cu and let torch's hipify handle it.
            hip_stems = {p.stem for p in kernel_hip}
            for p in kernel_cu:
                if p.stem in hip_stems:
                    kernel_sources.append(str(p.parent / (p.stem + '.hip')))
                else:
                    kernel_sources.append(str(p))
            # Standalone .hip files that have no .cu counterpart
            cu_stems = {p.stem for p in kernel_cu}
            for p in kernel_hip:
                if p.stem not in cu_stems:
                    kernel_sources.append(str(p))

            # --- .cpp files ---
            # When torch's hipify finds a source whose hipified version
            # already exists, it sets hipified_path=None which crashes ninja.
            # Fix: detect paired original ↔ hip files and keep only the hip one.
            cpp_names = {p.name: p for p in kernel_cpp}
            all_cpp_names = set(cpp_names.keys())
            skip = set()
            for name in all_cpp_names:
                if name in skip:
                    continue
                hip_name = _has_hip_counterpart(name, all_cpp_names)
                if hip_name:
                    skip.add(name)       # drop original
            for name, p in cpp_names.items():
                if name not in skip:
                    kernel_sources.append(str(p))
        else:
            # CUDA builds: ignore HIP-only shim files.
            selected_cpp = [p for p in kernel_cpp if not p.name.endswith('_hip.cpp')]
            kernel_sources = [str(p) for p in kernel_cu] + [str(p) for p in selected_cpp]

    return sorted(set(root_sources + kernel_sources))


def _patch_shape_to_sizes(sources: list[str]) -> None:
    """Replace .shape (Python-style) with .sizes() in C++ binding sources."""
    pattern = re.compile(r'\.shape\b')
    for src in sources:
        p = Path(src)
        if not p.exists() or p.suffix not in ('.cpp', '.h', '.cuh'):
            continue
        text = p.read_text(errors='replace')
        if '.shape' not in text:
            continue
        new_text = pattern.sub('.sizes()', text)
        if new_text != text:
            p.write_text(new_text)
            print(f'  [patch] {src}: .shape -> .sizes()')


def compile_kernels() -> int:
    build_dir = Path('build/forced_compile')
    output_so = Path('cuda_extension.so')
    backend = detect_backend()
    sources = find_sources(backend)

    if backend == 'none':
        print('Error: this PyTorch build has neither CUDA nor HIP backend.')
        return 1

    if not sources:
        print('Error: no source files found (*.cu, *.cpp in root or kernels/)')
        return 1

    print(f'Backend: {backend}')
    print(f'Compiling {len(sources)} files: {", ".join(sources)}')

    # Fix: .shape → .sizes() in C++ sources (common agent code-gen mistake)
    _patch_shape_to_sizes(sources)

    # Fix: don't rmtree build_dir — torch's file_baton creates a lock file
    # inside it; deleting it causes FileNotFoundError on release.  Instead
    # only clean object/so artefacts so ninja does a proper incremental rebuild.
    build_dir.mkdir(parents=True, exist_ok=True)

    if output_so.exists():
        output_so.unlink()

    # Include paths: always include cwd so that #include "../binding_registry.h"
    # from kernels/ can resolve when hipcc runs from the build directory.
    cwd = os.path.abspath('.')
    include_paths = [cwd, os.path.join(cwd, 'kernels')]

    try:
        if backend == 'hip':
            # Torch's hipify returns hipified_path=None for files it considers
            # "already hipified", crashing ninja with TypeError.
            # Fix: let hipify run normally but patch None → original source path.
            from torch.utils.hipify import hipify_python
            _real_hipify = hipify_python.hipify

            def _safe_hipify(**kw):
                result = _real_hipify(**kw)
                for path, hr in result.items():
                    if hr.hipified_path is None:
                        hr.hipified_path = path
                return result

            hipify_python.hipify = _safe_hipify
            try:
                module = cpp_ext.load(
                    name='cuda_extension',
                    sources=sources,
                    build_directory=str(build_dir),
                    verbose=True,
                    with_cuda=True,
                    extra_cflags=['-O3', '-std=c++17'],
                    extra_cuda_cflags=['-O3'],
                    extra_include_paths=include_paths,
                )
            finally:
                hipify_python.hipify = _real_hipify
        else:
            module = cpp_ext.load(
                name='cuda_extension',
                sources=sources,
                build_directory=str(build_dir),
                verbose=True,
                with_cuda=True,
                extra_cflags=['-O3', '-std=c++17'],
                extra_cuda_cflags=['-O3'],
                extra_include_paths=include_paths,
            )
    except Exception as exc:
        print('Compilation failed.')
        print(str(exc))
        traceback.print_exc()
        return 1

    built_so = Path(module.__file__)
    if built_so.exists():
        shutil.copy2(built_so, output_so)
        print(f'Compile success: {output_so}')
        return 0

    print('Compilation finished but cuda_extension.so was not generated.')
    return 1


def main() -> int:
    return compile_kernels()


if __name__ == '__main__':
    sys.exit(main())
