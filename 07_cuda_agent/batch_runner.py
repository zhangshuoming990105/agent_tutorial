#!/usr/bin/env python3
"""
Batch runner for 07_cuda_agent.

Runs multiple CUDA agent tasks in parallel, one agent per GPU at a time.
Uses a dynamic GPU pool so that:
  1. No two of our tasks ever run on the same GPU simultaneously (threading lock).
  2. GPUs already occupied by external jobs are skipped until they become idle
     (idle check via rocm-smi / nvidia-smi, same criterion as gpu_pool.acquire_gpu).

Workers are not bound to a fixed GPU — they acquire whichever GPU is free
at the moment a task starts, and release it when the task finishes.

The agent-internal logic (history loading, --max-agent-steps, etc.) is
completely orthogonal to this script — it only controls *how many* agents
run at once and *which GPU* each one uses.

Usage
-----
    python batch_runner.py 1 10                     # tasks 1–10 (global IDs, 1-based)
    python batch_runner.py --tasks "1-10"           # same
    python batch_runner.py --tasks "level1/1-10"   # level1 tasks 1–10
    python batch_runner.py --tasks "1,3,7,9"       # specific task IDs
    python batch_runner.py 1 10 --max-workers 2    # at most 2 parallel agents
    python batch_runner.py 1 10 --dry-run          # preview without running
"""

from __future__ import annotations

import argparse
import queue
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AGENT_DIR))


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

def detect_gpu_indices() -> list[str]:
    """Return ordered list of GPU indices found on this machine.

    Uses the same gpu_pool logic as chatbot.py. Falls back to ["0"] if
    detection fails so the runner is always usable.
    """
    try:
        from gpu_pool import query_gpus  # noqa: PLC0415
        gpus = query_gpus()
        if gpus:
            return [g["index"] for g in gpus]
    except Exception:
        pass

    # Direct nvidia-smi fallback
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            indices = [ln.strip() for ln in result.stdout.strip().splitlines() if ln.strip()]
            if indices:
                return indices
    except Exception:
        pass

    return ["0"]


# ---------------------------------------------------------------------------
# Dynamic GPU pool
# ---------------------------------------------------------------------------

class GpuPool:
    """Thread-safe GPU pool with idle checking.

    acquire() blocks until a GPU that is:
      - not currently held by one of our workers (threading.Lock), AND
      - not occupied by an external job (util==0%, mem<1% via nvidia/rocm-smi)
    is available, then returns its index string.

    release() must be called when the task finishes.
    """

    POLL_INTERVAL = 15.0   # seconds between retry sweeps when all GPUs are busy

    def __init__(self, gpu_indices: list[str]) -> None:
        self._gpus = gpu_indices
        # One non-reentrant lock per GPU; held for the entire duration of a task.
        self._locks: dict[str, threading.Lock] = {
            g: threading.Lock() for g in gpu_indices
        }

    def acquire(self) -> str:
        """Block until an idle, unlocked GPU is available. Returns its index."""
        warned = False
        while True:
            for gpu in self._gpus:
                lock = self._locks[gpu]
                # Non-blocking try — skip GPUs already held by our workers.
                if not lock.acquire(blocking=False):
                    continue
                # Lock acquired: now check whether the GPU is actually idle.
                if self._is_idle(gpu):
                    return gpu   # caller must call release(gpu)
                # GPU busy (external job) — release our lock and try next.
                lock.release()

            if not warned:
                print(
                    f"[{_ts()}] [pool] All GPUs busy, polling every "
                    f"{self.POLL_INTERVAL:.0f}s ...",
                    flush=True,
                )
                warned = True
            time.sleep(self.POLL_INTERVAL)

    def release(self, gpu: str) -> None:
        """Release a previously acquired GPU lock."""
        self._locks[gpu].release()

    @staticmethod
    def _is_idle(gpu: str) -> bool:
        """Return True if the GPU has 0% utilisation and <1% memory usage."""
        try:
            from gpu_pool import query_gpus  # noqa: PLC0415
            for g in query_gpus():
                if g["index"] == gpu:
                    return g["util"] == 0.0 and g["mem_pct"] < 1.0
            return True   # index not found — assume idle
        except Exception:
            return True   # can't query → assume idle rather than blocking forever


# ---------------------------------------------------------------------------
# Task spec parsing
# ---------------------------------------------------------------------------

def parse_tasks(spec: str) -> list[str]:
    """Parse a task specification string into a list of chatbot.py --task values.

    Supported formats (can be comma-separated):
      "1-10"              → ["1", "2", ..., "10"]   (global IDs, 1-based)
      "level1/1-10"       → ["level1/1", ..., "level1/10"]
      "1,3,5"             → ["1", "3", "5"]
      "level1/1,level1/5" → ["level1/1", "level1/5"]
    """
    tasks: list[str] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue

        # level1/N-M  range
        m = re.match(r"^(level\d+)/(\d+)-(\d+)$", part)
        if m:
            level, start, end = m.group(1), int(m.group(2)), int(m.group(3))
            tasks.extend(f"{level}/{i}" for i in range(start, end + 1))
            continue

        # N-M  plain integer range
        m = re.match(r"^(\d+)-(\d+)$", part)
        if m:
            start, end = int(m.group(1)), int(m.group(2))
            tasks.extend(str(i) for i in range(start, end + 1))
            continue

        # Single value (integer or level/N)
        tasks.append(part)

    return tasks


# ---------------------------------------------------------------------------
# Task execution
# ---------------------------------------------------------------------------

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def run_task(task_spec: str, gpu_index: str, extra_args: list[str],
             task_timeout: int = 1200) -> int:
    """Launch chatbot.py for one task on a dedicated GPU.

    stdout is suppressed here because chatbot.py already writes a full log
    to <task_dir>/logs/<timestamp>.log. stderr is captured and printed only
    on failure.

    stdin is closed (/dev/null) so the agent exits cleanly after finishing
    the initial autonomous task instead of blocking on interactive input.

    task_timeout: hard wall-clock limit in seconds (default 1200 = 20 min).
    If exceeded, the subprocess is killed and the task is marked TIMEOUT.
    This protects against: LLM API hanging, GPU kernel deadlocks, or any
    other scenario where chatbot.py stops making progress.
    """
    cmd = [
        sys.executable,
        str(AGENT_DIR / "chatbot.py"),
        "--task", task_spec,
        "--gpu", gpu_index,
        *extra_args,
    ]

    print(f"[{_ts()}] START   task={task_spec:<14s} gpu={gpu_index}", flush=True)

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(AGENT_DIR),
            stdin=subprocess.DEVNULL,   # agent exits on EOF after initial task
            stdout=subprocess.DEVNULL,  # chatbot.py logs to <task_dir>/logs/
            stderr=subprocess.PIPE,
            text=True,
            timeout=task_timeout,
        )
        rc = proc.returncode
        if rc != 0 and proc.stderr:
            for line in proc.stderr.strip().splitlines():
                print(f"  [stderr:{task_spec}] {line}", flush=True)
    except subprocess.TimeoutExpired as exc:
        exc.kill()  # ensure the hung subprocess is terminated
        print(
            f"[{_ts()}] TIMEOUT task={task_spec:<14s} gpu={gpu_index}  "
            f"(killed after {task_timeout}s)",
            flush=True,
        )
        rc = -2
    except Exception as exc:
        print(f"[{_ts()}] ERROR   task={task_spec:<14s} launch failed: {exc}", flush=True)
        rc = -1

    status = "OK" if rc == 0 else ("TIMEOUT" if rc == -2 else f"FAIL(exit={rc})")
    print(f"[{_ts()}] DONE    task={task_spec:<14s} gpu={gpu_index}  {status}", flush=True)
    return rc


# ---------------------------------------------------------------------------
# Task result reading
# ---------------------------------------------------------------------------

def read_task_result(task_spec: str) -> str:
    """Return a one-line result summary from the most recent history entry.

    Reads <task_dir>/history/<latest>/result.json after a task finishes.
    Returns an empty string if the file is missing or unreadable.
    """
    import json as _json
    try:
        from cuda_task import resolve_task_path  # noqa: PLC0415
        task_dir = resolve_task_path(task_spec)
        history_dir = task_dir / "history"
        if not history_dir.is_dir():
            return ""
        # Most recent entry = lexicographically last timestamp directory
        entries = sorted(
            (e for e in history_dir.iterdir() if e.is_dir()),
            key=lambda e: e.name,
            reverse=True,
        )
        for entry in entries:
            result_file = entry / "result.json"
            if result_file.is_file():
                data = _json.loads(result_file.read_text(encoding="utf-8"))
                verify = data.get("verify", "?")
                profile = data.get("profile") or {}
                if profile:
                    baseline = profile.get("baseline_us", 0.0)
                    cuda = profile.get("cuda_us", 0.0)
                    compile_t = profile.get("compile_us", 0.0)
                    speedup = baseline / cuda if cuda > 0 else 0.0
                    return (
                        f"verify={verify}  "
                        f"baseline={baseline:.0f}us  "
                        f"torch.compile={compile_t:.0f}us  "
                        f"cuda={cuda:.0f}us  "
                        f"speedup={speedup:.2f}x"
                    )
                return f"verify={verify}  (no profile data)"
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------------------
# Progress tracker
# ---------------------------------------------------------------------------

class ProgressTracker:
    """Thread-safe tracker that prints a live summary after each task completes."""

    def __init__(self, total: int) -> None:
        self.total = total
        self._lock = threading.Lock()
        self._ok: list[dict] = []
        self._failed: list[dict] = []
        self._inflight: dict[str, str] = {}   # task_spec → gpu

    def task_started(self, task_spec: str, gpu: str) -> None:
        with self._lock:
            self._inflight[task_spec] = gpu

    def task_done(self, task_spec: str, gpu: str, rc: int, result_line: str = "") -> None:
        with self._lock:
            self._inflight.pop(task_spec, None)
            entry = {"task": task_spec, "gpu": gpu, "rc": rc}
            if rc == 0:
                self._ok.append(entry)
            else:
                self._failed.append(entry)
            if result_line:
                print(f"           {result_line}", flush=True)
            self._print_status_locked()

    def _print_status_locked(self) -> None:
        completed = len(self._ok) + len(self._failed)
        inflight_str = (
            "  In-flight: " + " ".join(
                f"{t}(g{g})" for t, g in sorted(self._inflight.items())
            )
            if self._inflight else ""
        )
        bad = [r for r in self._failed if r["rc"] != -2]
        tout = [r for r in self._failed if r["rc"] == -2]
        bad_str = ("  Failed: " + " ".join(r["task"] for r in bad)) if bad else ""
        tout_str = ("  Timeout: " + " ".join(r["task"] for r in tout)) if tout else ""
        bar = f"[{_ts()}] ── {completed}/{self.total} done"
        parts = [bar, f"OK={len(self._ok)}", f"Failed={len(bad)}", f"Timeout={len(tout)}"]
        print("  ".join(parts) + inflight_str + bad_str + tout_str, flush=True)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def worker(
    pool: GpuPool,
    task_queue: "queue.Queue[str]",
    results: list[dict],
    lock: threading.Lock,
    extra_args: list[str],
    progress: ProgressTracker,
    task_timeout: int,
) -> None:
    """Drain *task_queue*, acquiring a free GPU from the pool for each task."""
    while True:
        try:
            task_spec = task_queue.get_nowait()
        except queue.Empty:
            break

        gpu = pool.acquire()          # blocks until a free, idle GPU is available
        progress.task_started(task_spec, gpu)
        try:
            rc = run_task(task_spec, gpu, extra_args, task_timeout=task_timeout)
        finally:
            pool.release(gpu)         # always release, even if run_task raises

        # Read verify/profile result written by chatbot.py before reporting progress
        result_line = read_task_result(task_spec) if rc == 0 else ""
        progress.task_done(task_spec, gpu, rc, result_line)

        with lock:
            results.append({"task": task_spec, "gpu": gpu, "rc": rc})

        task_queue.task_done()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Batch runner for 07_cuda_agent — runs tasks in parallel, "
            "one agent per GPU."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python batch_runner.py 1 10                   # tasks 1–10 (global IDs, 1-based)
  python batch_runner.py --tasks "1-10"         # same
  python batch_runner.py --tasks "level1/1-10"  # level1 tasks 1–10
  python batch_runner.py --tasks "1,3,5,7"      # specific global IDs
  python batch_runner.py 1 10 --max-workers 2   # at most 2 parallel agents
  python batch_runner.py 1 10 --dry-run         # preview plan without running

note: global task IDs are 1-based (level1 = 1–99, level2 = 100–198, ...).
        """,
    )
    parser.add_argument(
        "start", nargs="?", type=int,
        help="Start task ID (global integer, 1-based, inclusive)",
    )
    parser.add_argument(
        "end", nargs="?", type=int,
        help="End task ID (global integer, inclusive)",
    )
    parser.add_argument(
        "--tasks", type=str,
        help='Task spec string: "1-10", "level1/1-10", "1,3,5"',
    )
    parser.add_argument(
        "--max-workers", type=int, default=None,
        help="Max parallel agents (default: number of GPUs detected; capped at GPU count)",
    )
    parser.add_argument(
        "--max-agent-steps", type=int, default=30,
        help="Max autonomous steps per task, passed to chatbot.py (default: 30)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=128_000,
        help="Context window size, passed to chatbot.py (default: 128000)",
    )
    parser.add_argument(
        "--task-timeout", type=int, default=1200,
        help="Hard wall-clock timeout per task in seconds (default: 1200 = 20 min). "
             "If a task (chatbot.py process) does not finish within this time it is "
             "killed and marked as TIMEOUT. Protects against LLM API hangs, GPU kernel "
             "deadlocks, and any other runaway subprocess.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would run without actually launching any agents",
    )

    args = parser.parse_args()

    # ---- Build task list ----
    if args.start is not None and args.end is not None:
        if args.start < 1:
            parser.error("Task IDs are 1-based; start must be >= 1.")
        tasks = [str(i) for i in range(args.start, args.end + 1)]
    elif args.tasks:
        tasks = parse_tasks(args.tasks)
    else:
        parser.error(
            "Specify tasks as positional arguments 'start end' "
            "or via --tasks."
        )

    if not tasks:
        print("[batch] No tasks to run.", flush=True)
        return

    # ---- GPU detection ----
    gpu_indices = detect_gpu_indices()
    num_gpus = len(gpu_indices)

    # ---- Worker count ----
    # Workers compete for GPUs dynamically via the pool, so max_workers just
    # caps how many tasks can be *in-flight* at once. Capping at num_gpus is
    # the natural upper bound (no point having more workers than GPUs).
    requested = args.max_workers if args.max_workers is not None else num_gpus
    if requested > num_gpus:
        print(
            f"[batch] Warning: --max-workers={requested} exceeds detected GPU count "
            f"({num_gpus}). Capping at {num_gpus}.",
            flush=True,
        )
        requested = num_gpus
    max_workers = max(1, min(requested, len(tasks)))

    # ---- Extra args forwarded to chatbot.py ----
    extra_args = [
        "--max-agent-steps", str(args.max_agent_steps),
        "--max-tokens", str(args.max_tokens),
    ]

    # ---- Announce plan ----
    print(f"[batch] GPUs in pool  : {gpu_indices}")
    print(f"[batch] Tasks         : {len(tasks)}")
    print(f"[batch] Workers       : {max_workers}  (dynamic GPU pool — idle check per task)")
    print(f"[batch] Task list     : {tasks}")

    if args.dry_run:
        print("\n[dry-run] Would run (GPU assigned dynamically at task start):")
        for task in tasks:
            print(f"  task={task}")
        print(
            f"\n  Up to {max_workers} tasks run in parallel; each waits for a free "
            f"GPU (util=0%, mem<1%) before starting."
        )
        return

    print(flush=True)

    # ---- Build dynamic GPU pool and progress tracker ----
    pool = GpuPool(gpu_indices)
    progress = ProgressTracker(total=len(tasks))

    # ---- Run ----
    task_q: queue.Queue[str] = queue.Queue()
    for t in tasks:
        task_q.put(t)

    results: list[dict] = []
    lock = threading.Lock()
    batch_start = time.time()

    threads = [
        threading.Thread(
            target=worker,
            args=(pool, task_q, results, lock, extra_args, progress, args.task_timeout),
            daemon=True,
            name=f"worker-{i}",
        )
        for i in range(max_workers)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    elapsed = time.time() - batch_start

    # ---- Summary ----
    ok = [r for r in results if r["rc"] == 0]
    timed_out = [r for r in results if r["rc"] == -2]
    failed = [r for r in results if r["rc"] not in (0, -2)]

    print("\n" + "=" * 62)
    print(f"BATCH COMPLETE  ({elapsed / 60:.1f} min  /  {elapsed:.0f}s)")
    print("=" * 62)
    print(f"Total: {len(results)}   OK: {len(ok)}   Failed: {len(failed)}   Timeout: {len(timed_out)}")
    print()
    for r in sorted(results, key=lambda x: x["task"]):
        if r["rc"] == 0:
            status = "OK"
        elif r["rc"] == -2:
            status = f"TIMEOUT(>{args.task_timeout}s)"
        else:
            status = f"FAIL(exit={r['rc']})"
        print(f"  task={r['task']:<14s}  gpu={r['gpu']}  {status}")
    if failed:
        print(f"\nFailed tasks: {[r['task'] for r in failed]}")
    if timed_out:
        print(f"Timed-out tasks: {[r['task'] for r in timed_out]}")
    if failed or timed_out:
        print("Check individual logs in <task_dir>/logs/ for details.")


if __name__ == "__main__":
    main()
