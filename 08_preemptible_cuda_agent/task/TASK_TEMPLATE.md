# CUDA Codegen Task Template (General)

This is a **generic task card** for CUDA/HIP code generation tasks in `08_preemptible_cuda_agent`.
Use it when a case directory does not provide its own `task.md`.

## 1) Scope and Objective

- Input: one case directory that contains `model.py`.
- Goal: produce a correct and performant CUDA/HIP extension implementation.
- Deliverables:
  - generated kernel sources in `kernels/` (`.cu` and/or `.hip`)
  - binding source in `kernels/*_binding.cpp` (or `*_binding_hip.cpp` if project uses it)
  - `model_new.py` that calls the extension path
  - profiling evidence from `utils/profiling.py`

## 2) Project Structure Contract

Within case `workdir`:

- fixed infrastructure (do not modify unless explicitly required):
  - `utils/`
  - `binding.cpp`
  - `binding_registry.h`
- files to read first:
  - `model.py`
- files expected to be generated/updated:
  - `kernels/`
  - `model_new.py`

## 3) Workspace and Entry

Agent should run inside task workdir.

Entry examples:

```bash
python chatbot.py --task level1/003
python chatbot.py --task /abs/path/to/case_dir
python chatbot.py --task 42
```

Chat-first mode:

```text
/task load <specifier>
/task inject
```

## 4) Standard Workflow (Must Follow)

1. Read and understand `model.py`:
   - constructor signature
   - forward path
   - input shape/dtype characteristics
2. Generate extension implementation:
   - kernel source under `kernels/`
   - corresponding binding under `kernels/`
   - `model_new.py` invoking extension op(s)
3. Compile:
   - `bash utils/compile.sh`
4. Verify correctness:
   - `python3 -m utils.verification`
5. Profile performance:
   - `python3 -m utils.profiling`
6. If needed, iterate on implementation and re-run compile/verify/profile.

## 5) Commands

Run from case workdir:

```bash
bash utils/compile.sh
python3 -m utils.verification
python3 -m utils.profiling
```

Optional operator attribution:

```bash
/opt/dtk/bin/hipprof --stats --hip-trace -o op_profile python3 -m utils.profiling
```

## 6) Acceptance Criteria

- verification passes
- extension path is actually exercised in `model_new.py`
- profiling output includes:
  - `Torch Baseline`
  - `Torch Compile`
  - `CUDA Extension`
- generated code is reproducible from workdir commands above

## 7) Notes

- If a case-specific `task.md` exists, it should override this template.
- If none exists, this template is treated as the default runbook.
