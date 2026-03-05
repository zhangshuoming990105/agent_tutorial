#!/usr/bin/env python3
"""
One-shot script to extract a clean task dataset from CUDA-Agent results.

Reads metadata.json from each result folder, extracts the pytorch_module
(model.py content) and metadata, and writes a clean dataset with numeric IDs.

Usage:
    python build_dataset.py [--source /path/to/CUDA-Agent/results] [--output dataset]
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

DEFAULT_SOURCE = Path(__file__).resolve().parent.parent.parent / "cuda-agent-base" / "CUDA-Agent" / "results"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "dataset"
LEVELS = ["level1", "level2", "level3"]


def extract_description(module_source: str) -> str:
    """Extract the Model class docstring as a one-line description."""
    lines = module_source.split("\n")
    in_class = False
    in_docstring = False
    for line in lines:
        if "class Model" in line:
            in_class = True
            continue
        if in_class and not in_docstring:
            stripped = line.strip()
            if stripped.startswith('"""') or stripped.startswith("'''"):
                quote = stripped[:3]
                rest = stripped[3:]
                if rest.endswith(quote) and len(rest) > 3:
                    return rest[:-3].strip()
                if rest.strip():
                    return rest.strip()
                in_docstring = True
                continue
            break
        if in_docstring:
            stripped = line.strip()
            if '"""' in stripped or "'''" in stripped:
                cleaned = stripped.replace('"""', "").replace("'''", "").strip()
                if cleaned:
                    return cleaned
                break
            if stripped:
                return stripped
    return ""


def build_dataset(source: Path, output: Path) -> None:
    if not source.is_dir():
        print(f"Error: source directory not found: {source}")
        sys.exit(1)

    index: dict[str, dict] = {}
    total = 0

    for level in LEVELS:
        level_dir = source / level
        if not level_dir.is_dir():
            print(f"Warning: {level_dir} not found, skipping")
            continue

        uuids = sorted(
            d for d in os.listdir(level_dir)
            if (level_dir / d).is_dir()
        )

        print(f"{level}: {len(uuids)} tasks")

        for i, uuid in enumerate(uuids, start=1):
            task_id = f"{level}/{i:03d}"
            task_dir = level_dir / uuid
            metadata_file = task_dir / "metadata.json"

            module_source = None
            metadata = {}

            if metadata_file.is_file():
                try:
                    with open(metadata_file, encoding="utf-8") as f:
                        metadata = json.load(f)
                    module_source = metadata.get("pytorch_module")
                except (json.JSONDecodeError, OSError):
                    pass

            if not module_source:
                model_file = task_dir / "model.py"
                if not model_file.is_file():
                    model_file = task_dir / "workdir" / "model.py"
                if model_file.is_file():
                    module_source = model_file.read_text(encoding="utf-8", errors="replace")

            if not module_source:
                print(f"  Warning: no model source for {uuid}, skipping")
                continue

            out_dir = output / level / f"{i:03d}"
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "model.py").write_text(module_source, encoding="utf-8")

            description = extract_description(module_source)

            score = metadata.get("score", "")
            try:
                speedup_compile = float(metadata.get("speedup_torch_compile", 0))
            except (ValueError, TypeError):
                speedup_compile = 0.0
            try:
                speedup_baseline = float(metadata.get("speedup_torch_baseline", 0))
            except (ValueError, TypeError):
                speedup_baseline = 0.0

            index[task_id] = {
                "uuid": uuid,
                "description": description,
                "score": score,
                "speedup_torch_compile": round(speedup_compile, 2),
                "speedup_torch_baseline": round(speedup_baseline, 2),
            }
            total += 1

    index_file = output / "index.json"
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"\nDataset built: {total} tasks -> {output}")
    print(f"Index: {index_file}")


def main():
    parser = argparse.ArgumentParser(description="Build clean CUDA task dataset")
    parser.add_argument(
        "--source", type=str, default=str(DEFAULT_SOURCE),
        help=f"Path to CUDA-Agent results directory (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--output", type=str, default=str(DEFAULT_OUTPUT),
        help=f"Output dataset directory (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()
    build_dataset(Path(args.source), Path(args.output))


if __name__ == "__main__":
    main()
