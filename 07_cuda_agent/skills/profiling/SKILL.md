---
name: profiling
description: Profiles GPU operators on ROCm or Hygon DCU using hipprof, MIOpen logs, and vendor driver replay. Use when user asks for profiling, bottleneck analysis, kernel attribution, solver identification, or finding SOTA backend implementations for any operator.
tools:
  - run_shell
  - read_file
  - write_file
  - search_files
triggers:
  - profile
  - profiling
  - bottleneck
  - kernel
  - hipprof
  - rocprof
  - miopen
  - operator
  - 算子
  - 性能分析
always_on: false
---

# ROCm/Hygon Profiling Skill

## Goal

For any operator (`conv`, `matmul`, `attention`, custom op), follow this fixed flow:

1. **Measure** true runtime path
2. **Attribute** time to concrete kernels/APIs
3. **Identify** backend solver/algorithm
4. **Locate** implementation source (local + upstream web/repo)
5. **Reproduce** with vendor driver command
6. **Report** actionable optimization direction

Do not tune code before steps 1-4 are complete.

## Step 1: Baseline timing (user workload)

Use workload-native timing first (PyTorch events or existing benchmark script).

- Record:
  - shape
  - dtype
  - warmup/iters
  - avg latency

If Python overhead is suspected, add a direct extension/operator-only timing run.

## Step 2: Kernel/API attribution with hipprof

### Fast attribution

```bash
/opt/dtk/bin/hipprof --stats --hip-trace -o <run_name> <app_cmd>
```

Collect from output files:

- `<run_name>.kernel.csv` (kernel names + durations)
- `<run_name>.hiptrace.csv` (API timeline)

### Kernel-specific counters

```bash
/opt/dtk/bin/hipprof --pmc --pmc-type 3 --kernel-name "<kernel_substr>" <app_cmd>
```

Use this to compare two implementations on the same shape.

## Step 3: Backend solver identification (MIOpen path)

When workload uses MIOpen:

```bash
MIOPEN_ENABLE_LOGGING=1 \
MIOPEN_ENABLE_LOGGING_CMD=1 \
MIOPEN_LOG_LEVEL=6 \
<app_cmd>
```

Extract:

- algorithm enum (for example `miopenConvolutionBwdDataAlgoWinograd`)
- solver id/name (for example `HConv677`)
- emitted `MIOpenDriver` command

Treat these logs as ground truth over framework labels.

## Step 4: Replay with vendor driver

Run the emitted command directly (same shape/params):

```bash
/opt/dtk/bin/MIOpenDriver conv ...
```

This isolates backend kernel performance from framework overhead.

## Step 5: Online SOTA implementation discovery strategy

This is the required "联网找 SOTA" strategy:

1. **Profile first** to get real kernel/solver id.
2. **Inspect local toolchain** (`/opt/dtk`, headers, db, libs) for available clues.
3. **Search docs/web** for solver family and command semantics.
4. **Inspect upstream source repo** (typically `ROCm/MIOpen`):
   - map solver selection file
   - map kernel wrapper asm
   - map included code body and metadata
5. **Cross-check** by replaying driver command and matching performance behavior.

Never start from blind web keyword guessing without runtime attribution data.

## Step 6: Source mapping heuristics (MIOpen Winograd example)

Typical mapping:

- solver selection: `src/solver/conv/conv_winoRxS.cpp`
- kernel wrapper: `src/kernels/Conv_Winograd_*.s`
- kernel body: `src/kernels/Conv_Winograd_*.inc`
- args layout: `src/kernels/Conv_Winograd_*_metadata.inc`

The same pattern generalizes to other solver families.

## Step 7: Integration options

Prioritize in this order:

1. **MIOpen Immediate API integration** in extension (lowest risk, usually near vendor speed)
2. standalone code object + module launch
3. handwritten kernel rewrite (only when backend path is clearly unsuitable)

## Output template

Return results in this structure:

1. baseline latency
2. top kernels by time
3. chosen backend algorithm/solver
4. vendor driver replay latency
5. source file mapping (exact paths)
6. recommended next action (integrate/tune/rewrite)

## Artifact hygiene

After profiling sessions, clean generated artifacts from project root/workdir:

- `pmc_results_*.csv`
- `<run_name>.kernel.csv`
- `<run_name>.hiptrace.csv`
- `<run_name>.db`

Keep only files explicitly needed for reproducibility.

## Reference examples

- Conv Winograd one-shot example:
  - `../conv/examples/conv_transpose2d_winograd_hconv677.hip`
