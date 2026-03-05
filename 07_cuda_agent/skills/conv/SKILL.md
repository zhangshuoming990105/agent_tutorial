---
name: conv
description: Optimizes Conv/ConvTranspose kernels on ROCm or Hygon DCU by first profiling the real backend kernel, then locating upstream MIOpen/ROCm implementations via web+repo search, and finally integrating the best path into PyTorch CUDA/HIP extensions. Use when user mentions conv, conv2d, conv_transpose, winograd, miopen, rocprof, hipprof, or asks for SOTA kernel strategy.
---

# Conv Optimization Skill (ROCm/Hygon)

## Scope

Use this skill for:

- `Conv2d` / `ConvTranspose2d` / depthwise conv optimization tasks.
- Cases where the user asks to "beat PyTorch" or "align with MIOpen speed".
- "Find SOTA kernel implementation" requests that require both profiling and internet research.

Do **not** start by rewriting kernels blindly. Always identify what PyTorch actually launches first.

## Default Strategy: Profile -> Identify -> Locate Source -> Integrate

### 1) Profile the real kernel path

1. Run the target workload with GPU timing (torch events or driver timing).
2. Use `hipprof --stats --hip-trace` to extract kernel names and API time.
3. If backend is MIOpen, enable logs:
   - `MIOPEN_ENABLE_LOGGING=1`
   - `MIOPEN_ENABLE_LOGGING_CMD=1`
   - `MIOPEN_LOG_LEVEL=6` (or 7 for deeper traces)
4. Record:
   - solver name (for example `HConv677`)
   - algorithm (`miopenConvolutionBwdDataAlgoWinograd`, etc.)
   - shape signature

Never assume printed labels like "cuDNN" are accurate on ROCm/Hygon.

### 2) Reproduce with vendor driver

From MIOpen logs, extract the equivalent `MIOpenDriver conv ...` command and run it directly.

Goal:

- confirm the same algorithm/solver
- get clean reference kernel time without Python overhead

### 3) Locate source implementation (web + local)

Use both:

1. Local toolchain inspection (`/opt/dtk`, headers, libs, db)
2. Upstream search (`ROCm/MIOpen` repository via web/repo clone)

Mapping pattern for Winograd-like solvers:

- solver selection logic: `src/solver/conv/conv_winoRxS.cpp`
- kernel wrapper asm: `src/kernels/Conv_Winograd_*.s`
- included body: `src/kernels/Conv_Winograd_*.inc`
- metadata / args layout: `src/kernels/Conv_Winograd_*_metadata.inc`

### 4) Integrate into extension

Prefer **MIOpen Immediate API** integration first (lower risk, closest to vendor speed):

- `miopenConvolutionBackwardDataGetSolutionCount/GetSolution`
- select Winograd solution (or force by solver name if needed)
- `miopenConvolutionBackwardDataCompileSolution`
- `miopenConvolutionBackwardDataImmediate`

Only build standalone HSACO and manual `hipModuleLaunchKernel` when necessary.

### 5) Verify correctness and speed

Always run:

1. correctness checks (`torch.testing.assert_close`)
2. baseline comparison (`torch`, `torch.compile`, extension)
3. confirm selected solver/algorithm in logs

Report both kernel-time and end-to-end time.

## "联网找 SOTA 实现" Checklist

Copy and execute this checklist:

- [ ] Capture real kernel names via `hipprof`
- [ ] Capture MIOpen solver/algorithm via logging
- [ ] Reproduce with `MIOpenDriver`
- [ ] Map solver -> upstream source files
- [ ] Build extension path that calls the vendor-selected solver
- [ ] Verify correctness (multi-run)
- [ ] Compare performance vs PyTorch baseline
- [ ] Keep final reproducible commands in notes

## One-Shot Example (from real run)

Use this file as a one-shot integration reference:

- `examples/conv_transpose2d_winograd_hconv677.hip`

What it demonstrates:

- extension-side MIOpen handle/descriptor lifecycle
- selecting and compiling solution id for target shape
- immediate execution path for ConvTranspose2d-equivalent backward-data

## Practical Guidance

- For large spatial standard conv/transposed-conv, handwritten scalar kernels usually lose to MIOpen Winograd/ImplicitGEMM paths.
- For depthwise or unusual fused ops, custom kernels may still win.
- If extension is slower than expected, check for:
  - wrong class path being executed (stale model file, duplicate class definitions)
  - Python overhead dominating (validate with direct extension call benchmark)
  - wrong solution selected (inspect logs for `solution_id` and algorithm)

## Anti-Patterns

- Starting from tile-size tuning before identifying backend kernel.
- Comparing only Python-level timing without kernel-level confirmation.
- Treating profile alias names as exact source symbol names.
- Claiming SOTA without reproducing vendor-driver command path.
