# CUDA Agent — Task Outcome Analysis

This document describes the outcomes observed when running dataset tasks through
the CUDA agent system (`chatbot.py` / `batch_runner.py`) on the Hygon K100 (gfx928).

---

## Hardware context

| Property | Value |
|---|---|
| GPU | Hygon K100 AI (gfx928, ROCm/HIP) |
| Wavefront size | 64 lanes (not 32) |
| Shuffle primitives | `__shfl_down(val, offset)` — no mask argument |
| Ballot | `__ballot(pred)` — no mask argument |
| Memory | 64 GB HBM |

All `.cu` files are hipified and compiled with `hipcc` via `torch.utils.cpp_extension`.

---

## Typical task lifecycle

```
batch_runner  →  pool.acquire(gpu)          # wait for idle GPU
              →  chatbot.py --task N --gpu G
                    ├─ read model.py
                    ├─ write kernels/*.cu + *_binding.cpp
                    ├─ write model_new.py
                    ├─ bash utils/compile.sh          (~14 s per compile)
                    ├─ python3 -m utils.verification  (~5–30 s)
                    │    └─ PASS → auto-save to history + run profiling
                    │         └─ save profile to history/<timestamp>/result.json
                    ├─ (optional) rewrite + re-compile + re-verify + re-profile
                    └─ exit (EOF on stdin)
              →  pool.release(gpu)
              →  read_task_result() → best speedup across all history entries
```

---

## Outcome categories

### ① Quick win  (2–5 min)
Verification passes on the first attempt; profiling shows speedup ≥ 1.0×.

**Typical operations**: depthwise convolution, simple elementwise, ReLU, softmax.  
**Examples (level1)**: task 1 (depthwise conv, 1.91×), task 6 (1.19×), task 7 (1.00×).  
**Agent behaviour**: writes kernel → compile → verify → profile → stops.

---

### ② Correct but slow — one rewrite (5–10 min)
First profiling shows speedup < 0.9× vs `torch.compile`. The stopping rule
triggers one rewrite pass. The second attempt may or may not improve things.

**Typical operations**: anything where the first attempt uses too many threads,
poor memory layout, or forgot warp-size = 64.  
**Examples**: many level1 tasks on the first run (before the AMD hardware
section was added to the system prompt).  
**Agent behaviour**: write → compile → verify → profile (slow) → rewrite →
compile → verify → profile → stop.  
**History**: both profiling results are saved; `read_task_result` surfaces
the best one.

---

### ③ Correct but fundamentally limited (5–12 min)
Verification passes repeatedly but all profiling results show speedup ≪ 1×.
This happens for operations where PyTorch already calls vendor-optimised
libraries (rocBLAS, MIOpen) and writing a raw kernel cannot compete.

**Typical operations**:
- **Large GEMM** (e.g. 8205 × 2949 × 5921): PyTorch uses rocBLAS GEMM;
  any handwritten tiled kernel is orders of magnitude slower.
- **ConvTranspose2d**: scatter-add algorithm is inherently bandwidth-limited;
  a naive kernel runs ~0.05–0.10× of PyTorch.
- **Large-scale Conv2d**: similar story to GEMM.

**Agent behaviour**: multiple rewrite loops (up to step limit), each producing
a "correct but slow" result that is saved to history.  
**Mitigation**: all intermediate results are archived; the best one is reported.
Further improvement requires algorithmic breakthroughs (e.g. Winograd, im2col
with custom GEMM) or accepting the limitation.

---

### ④ Step limit reached with intermediate saves (10–20 min)
The agent exhausts `--max-agent-steps` (default 30) before writing a final
summary message. This can happen for:
- Algorithms with many debugging iterations (multi-block prefix scan / cumsum).
- Operations that produce subtle correctness failures that take many attempts to fix.

**What happens**: the agent loop exits, chatbot.py gets EOF on stdin and exits
cleanly. **All profiling results that occurred before the step limit are already
saved to history** (saved eagerly on every profiling tool result, not only on
the final assistant message). `batch_runner` reports `OK` (exit code 0) and
shows the best archived result.

**Example (level1/4, cumsum)**:
- Attempt 1: cuda = 518 ms  (0.08×)
- Attempt 2: cuda =  44 ms  (0.95×)  ← best, saved and reported
- Attempt 3: cuda = 128 ms  (0.33×)  ← regression, also saved
- Step limit hit → exit

---

### ⑤ GPU pool waiting
If all GPUs are occupied by external jobs when a worker tries to start a task,
`GpuPool.acquire()` polls every 15 s and prints a waiting message.
The worker resumes as soon as any GPU reports util = 0% and mem < 1%.

No tasks are lost; the queue simply waits.

---

### ⑥ Timeout (rare for level1)
If a task does not finish within `--task-timeout` seconds (default 1200 s = 20 min),
the subprocess is killed and the task is reported as `TIMEOUT`.

Causes:
- LLM API unresponsive (network issue).
- GPU kernel in an infinite loop during verification/profiling (mitigated by
  the 600 s timeout on individual `run_shell` calls, but the overall wall-clock
  limit provides a final safety net).
- Extremely long compilation (rare; typical compile is ~14 s).

Any history entries saved before the timeout are preserved and queryable.

---

## Performance summary — level1 tasks 1–10 (first run)

| Task | Operation | Speedup (best) | Category |
|------|-----------|----------------|----------|
| 1 | Depthwise conv | 1.91× | ① Quick win |
| 2 | Large GEMM | 0.12× | ③ Fundamentally limited |
| 3 | ConvTranspose2d | 0.06× | ③ Fundamentally limited |
| 4 | Cumsum | 0.95× | ④ Step limit, best saved |
| 5 | GEMM variant | 0.13× | ③ Fundamentally limited |
| 6 | (misc conv) | 1.19× | ① Quick win |
| 7 | (misc op) | 1.00× | ① Quick win |
| 8 | Depthwise-sep conv | 0.29× | ② Correct but slow |
| 9 | Conv2d large | 0.22× | ③ Fundamentally limited |
| 10 | (misc) | 0.82× | ② Correct but slow |

Batch wall-clock time: **~12 min** for 10 tasks on 8 GPUs.

---

## Key design decisions

| Decision | Rationale |
|---|---|
| One agent per GPU (GpuPool lock + idle check) | Prevents GPU contention between our agents and external jobs |
| Eager history save on every profiling tool result | Captures best intermediate result even when agent regresses or hits step limit |
| `read_task_result` picks best speedup across all history entries | A later attempt may be worse; always report the peak |
| Hard timeout per task (default 1200 s) | Prevents indefinite hang from LLM API or GPU kernel deadlock |
| `stdin = /dev/null` on subprocess | Agent exits cleanly after initial task without blocking on interactive input |

---

## Known limitations and future work

1. **GEMM-heavy tasks**: raw HIP GEMM cannot match rocBLAS. Possible future
   work: use `__hip_hgemm` intrinsics, or implement a Winograd-like
   convolution-specific optimisation.

2. **ConvTranspose2d**: consider an implicit GEMM approach or col2im + GEMM
   pipeline rather than a gather/scatter kernel.

3. **`verify_and_profile` as a single internal tool**: currently the agent
   calls verification and profiling as separate shell commands. Wrapping these
   into a single registered tool would remove the risk of the agent forgetting
   to profile, guarantee automatic history saving, and simplify the workflow.
   (Planned for next iteration.)

4. **Multi-run best tracking**: `read_task_result` already picks the best
   result across all history entries from all runs. Successive runs that improve
   on the previous best are automatically reflected.
