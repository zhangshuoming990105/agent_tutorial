"""
08_preemptible_cuda_agent - Preemptible CUDA kernel development agent.

Builds on Step 06's agent loop (error recovery, tool use, skill routing) and
specialises it for CUDA kernel development: given a PyTorch model.py the agent
writes CUDA kernels, compiles, verifies correctness, and profiles performance
in an autonomous compile-verify-profile loop.

New in this step:
- CUDA-specific system prompt with workspace structure and restrictions
- Task workspace initialisation (--task, --workdir)
- High-autonomy agent loop (default 20 steps) for compile-verify-profile cycles
- CUDA-aware failure classification and recovery nudges
- Shell safety OFF by default (compile/verify/profile are known-safe)
- Structured parsing of compilation, verification, and profiling outputs
- Queue-based runtime preemption of autonomous turns
"""

import argparse
from dataclasses import dataclass, field
from datetime import datetime
import json
from queue import Empty, Queue
import re
import select
import sys
import threading
from pathlib import Path
from typing import Any

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm import create_client, list_models  # noqa: E402

from compactor import compact_messages
from context import ContextManager
from cuda_task import (
    list_tasks,
    load_history_prompt,
    load_task_prompt,
    resolve_task_path,
    save_to_history,
    setup_workspace,
    workspace_summary,
)
from skill_manager import build_skill_prompt, load_skills, select_skills
from gpu_pool import gpu_status_summary
from runtime_state import (
    clear_preempt_request,
    is_autonomous_turn,
    request_preempt,
    set_autonomous_turn,
    set_shell_interrupt_on_preempt,
)
from tools import (
    execute_tool,
    get_all_tool_schemas,
    get_shell_policy_snapshot,
    get_tool_schema_map,
    set_gpu_auto,
    set_gpu_device,
    set_shell_safety,
    set_workspace_root,
    workspace_root_str,
)

DEFAULT_MAX_TOKENS = 128_000
DEFAULT_MAX_AGENT_STEPS = 30
QUEUE_EVENT_INPUT = "input"
QUEUE_EVENT_EOF = "eof"
QUEUE_EVENT_INTERRUPT = "interrupt"


_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE_RE.sub("", text)


class RunLogger:
    """Tee-writes to stdout (with ANSI colours) and a log file (plain text)."""

    def __init__(self, log_path: Path | None = None):
        self._file = None
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(log_path, "w", encoding="utf-8")

    def log(self, msg: str = "") -> None:
        print(msg)
        if self._file:
            self._file.write(_strip_ansi(msg) + "\n")
            self._file.flush()

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None


_logger = RunLogger()


def log(msg: str = "") -> None:
    _logger.log(msg)


class InputReaderThread(threading.Thread):
    """Reads stdin lines into a thread-safe queue for preemptible chat."""

    def __init__(self, user_queue: Queue[tuple[str, str]], stop_event: threading.Event):
        super().__init__(daemon=True)
        self._queue = user_queue
        self._stop_event = stop_event

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                ready, _, _ = select.select([sys.stdin], [], [], 0.2)
            except (ValueError, OSError):
                # stdin might be unavailable during shutdown.
                return
            if not ready:
                continue
            try:
                line = sys.stdin.readline()
            except KeyboardInterrupt:
                self._queue.put((QUEUE_EVENT_INTERRUPT, ""))
                return
            if line == "":
                self._queue.put((QUEUE_EVENT_EOF, ""))
                return

            text = line.strip()
            if not text:
                continue
            self._queue.put((QUEUE_EVENT_INPUT, text))

            # If the agent is in autonomous mode, this line requests preemption.
            if is_autonomous_turn():
                request_preempt()


def _dequeue_user_input_blocking(user_queue: Queue[tuple[str, str]]) -> tuple[str, str]:
    while True:
        event, text = user_queue.get()
        if event != QUEUE_EVENT_INPUT:
            return event, text
        if text.strip():
            return event, text.strip()


def _dequeue_user_input_nowait(user_queue: Queue[tuple[str, str]]) -> tuple[str, str] | None:
    while True:
        try:
            event, text = user_queue.get_nowait()
        except Empty:
            return None
        if event != QUEUE_EVENT_INPUT:
            return event, text
        if text.strip():
            return event, text.strip()


def _to_pending_input_from_preempt_event(
    preempt_item: tuple[str, str] | None,
    ctx: ContextManager,
) -> str | None:
    if not preempt_item:
        return None
    event, text = preempt_item
    if event in (QUEUE_EVENT_EOF, QUEUE_EVENT_INTERRUPT):
        return "quit"
    if event == QUEUE_EVENT_INPUT:
        ctx.messages.append(
            {
                "role": "system",
                "content": (
                    "## Runtime Preemption\n"
                    "Autonomous execution was preempted by new user input. "
                    "Prioritize the latest user message."
                ),
            }
        )
        return text.strip() if text else None
    return None


class _AsyncModelCall:
    def __init__(self) -> None:
        self.done = threading.Event()
        self.response: Any = None
        self.error: Exception | None = None


def _start_async_model_call(
    client: OpenAI,
    model: str,
    request_messages: list[dict],
    active_tool_schemas: list[dict],
) -> _AsyncModelCall:
    call = _AsyncModelCall()

    def worker() -> None:
        try:
            call.response = client.chat.completions.create(
                model=model,
                messages=request_messages,
                tools=active_tool_schemas,
            )
        except Exception as e:  # pragma: no cover - surfaced in main loop
            call.error = e
        finally:
            call.done.set()

    threading.Thread(target=worker, daemon=True).start()
    return call

_SYSTEM_PROMPT_TEMPLATE = """\
You are an expert CUDA kernel developer. Your task is to accelerate a PyTorch model \
by implementing custom CUDA C++ extensions that are both correct and fast.

Workspace root: {workspace_root}
All file-tool paths are relative to this root. run_shell executes in this directory by default.

## Hardware: AMD GPU (ROCm / HIP backend)
This system compiles .cu files via HIP (hipcc). Critical differences from NVIDIA CUDA:
- **Wavefront (warp) size = 64** — NOT 32. Always define `#define WARP_SIZE 64`.
- Warp shuffle: use `__shfl_down(val, offset)` — no mask argument (no `_sync` suffix).
- Ballot:       use `__ballot(pred)` — no mask argument.
- All warp-level loops must account for 64-lane wavefronts.

## Workspace Structure
```
.
├── binding_registry.h    # Do NOT modify - registration system
├── binding.cpp           # Do NOT modify - main module binding
├── kernels/              # YOUR WORK: implement all kernels here
├── utils/                # Do NOT modify - compile, verify, profile tools
├── model.py              # Do NOT modify - original PyTorch model
└── model_new.py          # YOUR WORK: optimized model using custom ops
```

## Critical Restrictions
- NO torch operators in C++ code (.cu or _binding.cpp files)
- NO torch.nn.functional operations in model_new.py — only tensor creation and custom ops
- NO library calls: no cuBLAS, cuDNN, MIOpen, or any external library. Write raw CUDA __global__ kernels only.
- NO modifications to utils/, binding.cpp, or binding_registry.h
- Work ONLY in kernels/ and model_new.py

## Workflow
1. Read model.py to understand the forward pass (check history/ if referenced above)
2. Write kernel .cu file + _binding.cpp file in kernels/
3. Write model_new.py using `import cuda_extension`
4. Compile: `bash utils/compile.sh`
5. Verify: `python3 -m utils.verification`
6. If compilation or verification fails, fix and re-run from step 4.
7. Once verification passes: run `python3 -m utils.profiling` once for measurement.
8. Apply the stopping rule below and output a summary.

## Stopping Rule
After the first successful profiling run, compare `CUDA Extension` time to `Torch Compile` time:

- **If CUDA Extension ≤ Torch Compile × 1.1** (your kernel is competitive): \
output a final summary and **STOP immediately**.
- **If CUDA Extension > Torch Compile × 1.1** (your kernel is slower): \
make **exactly one more rewrite** using a fundamentally different strategy \
(e.g. switch from naive to tiled GEMM, add vectorized loads, reduce register pressure, \
fix warp-size assumptions). Then re-compile → re-verify → re-profile once. \
Output a final summary and **STOP regardless** of the new result.

Do NOT attempt more than two full compile→verify→profile cycles. \
Do NOT write test scripts.

## User interaction policy
- Prefer autonomous execution for explicit action requests.
- Do NOT ask for routine permission to run compile/verify/profile steps.
- Ask the user only when genuinely blocked by missing critical inputs, conflicting constraints, \
or requests that may be risky/destructive.
- For conversational or non-action prompts, respond normally.

## Tool Usage Policy
1. Use tools directly — do not ask for permission or provide a plan first.
2. For file tasks: read_file -> write_file. For shell tasks: run_shell.
3. Keep responses concise; focus on action.
4. When compilation or verification fails, diagnose from tool output and fix immediately.
5. Do NOT create helper/test scripts. Only use the standard compile/verify/profile pipeline.
"""


def get_system_prompt() -> str:
    return _SYSTEM_PROMPT_TEMPLATE.format(workspace_root=workspace_root_str())


def compose_system_prompt(history_prompt: str = "", task_prompt: str = "") -> str:
    base = get_system_prompt()
    extras: list[str] = []
    if history_prompt:
        extras.append(history_prompt)
    if task_prompt:
        extras.append(task_prompt)
    if extras:
        base += "\n\n" + "\n\n".join(extras)
    return base


SLASH_COMMANDS_HELP = """Available commands:
  /task         - Show current task context status
  /task load X  - Load/switch task (resets conversation), e.g. /task load level1/003
  /task reload  - Reload current task context + history into system prompt (resets conversation)
  /task inject  - Inject current task.md/TASK.md into ongoing conversation
  /tokens       - Show token usage statistics
  /history      - Show all messages with token estimates (compact)
  /debug        - Show full context with readable summary (what the LLM sees)
  /debug raw    - Show raw OpenAI messages array (exact API payload, role/content/tool_calls)
  /compact [low|high] - Smart context compaction (default: low)
                    low  = conservative: keep structure, summarize redundant parts
                    high = aggressive: collapse everything to minimal bullet-list summaries
  /skills       - Show loaded skills and currently active skills
  /skill        - Pin/unpin a skill: /skill <name> on|off
  /verbose      - Show or set verbose token diagnostics: /verbose on|off
  /shell-safe   - Toggle safe shell mode: /shell-safe on|off
  /shell-policy - Show current shell allowlist/denylist
  /preempt      - Show preemption settings
  /preempt shell-kill on|off - Toggle killing long shell commands on preempt
  /recovery     - Show or set recovery cleanup: /recovery on|off
  /workspace    - Show current task workspace info
  /clear        - Reset conversation history
  /help         - Show this help message"""

CONTINUATION_REPLIES = {
    "y", "yes", "ok", "okay", "sure", "continue", "go", "proceed",
    "继续", "好的", "好",
}
ACTION_INTENT_RE = re.compile(
    r"\b(run|execute|check|list|read|write|search|find|show|compile|verify|profile|optimize|build|benchmark|fix|implement)\b|"
    r"(运行|执行|编译|验证|测速|性能|优化|实现|修改|修复)"
)
AUTONOMY_NUDGE_TEXT = (
    "Do not provide a plan or ask the user for procedural confirmation. "
    "For this turn, immediately call the required tools and only then provide a final summary."
)
EXIT_CODE_RE = re.compile(r"^exit_code=(-?\d+)$", re.MULTILINE)


# ---------------------------------------------------------------------------
# CUDA-specific output parsing
# ---------------------------------------------------------------------------

COMPILE_SUCCESS_RE = re.compile(r"Compile success", re.IGNORECASE)
COMPILE_FAIL_RE = re.compile(r"Compilation failed", re.IGNORECASE)
VERIFY_PASS_RE = re.compile(r"\[PASS\] verify success", re.IGNORECASE)
VERIFY_FAIL_RE = re.compile(r"(AssertionError|RuntimeError|Error|FAIL)", re.IGNORECASE)
PROFILE_RESULT_RE = re.compile(
    r"Torch Baseline:\s*([\d.]+)us.*Torch Compile:\s*([\d.]+)us.*CUDA Extension:\s*([\d.]+)us",
    re.IGNORECASE,
)


class CudaFailureKind:
    COMPILE = "compile"
    CORRECTNESS = "correctness"
    PERFORMANCE = "performance"
    GENERAL = "general"


def classify_cuda_failure(tool_name: str, result: str) -> str:
    """Classify a CUDA workflow failure for targeted recovery nudges."""
    if tool_name != "run_shell":
        return CudaFailureKind.GENERAL
    text = (result or "").strip()
    if COMPILE_FAIL_RE.search(text) or "error:" in text.lower():
        if any(kw in text.lower() for kw in ("undefined symbol", "nvcc", "hipcc", "syntax error",
                                               "no kernel image", "compilation failed")):
            return CudaFailureKind.COMPILE
    if "AssertionError" in text or "assert_close" in text or "RuntimeError" in text:
        return CudaFailureKind.CORRECTNESS
    if PROFILE_RESULT_RE.search(text):
        m = PROFILE_RESULT_RE.search(text)
        if m:
            try:
                compile_time = float(m.group(2))
                cuda_time = float(m.group(3))
                if cuda_time > compile_time * 0.95:
                    return CudaFailureKind.PERFORMANCE
            except ValueError:
                pass
    return CudaFailureKind.GENERAL


# ---------------------------------------------------------------------------
# Recovery state
# ---------------------------------------------------------------------------

@dataclass
class ToolExecutionOutcome:
    called: bool = False
    failures: list[str] = field(default_factory=list)
    failure_kinds: list[str] = field(default_factory=list)


@dataclass
class RecoveryState:
    had_failure: bool = False
    unresolved_failure: bool = False
    repeated_failure_count: int = 0
    last_failure_signature: str = ""
    failures: list[str] = field(default_factory=list)
    last_failure_kind: str = CudaFailureKind.GENERAL

    def record_failures(self, failures: list[str], kinds: list[str] | None = None) -> None:
        if not failures:
            self.unresolved_failure = False
            self.repeated_failure_count = 0
            return

        self.had_failure = True
        self.unresolved_failure = True
        signature = " | ".join(failures)
        if signature == self.last_failure_signature:
            self.repeated_failure_count += 1
        else:
            self.repeated_failure_count = 1
            self.last_failure_signature = signature
        self.failures.extend(failures[-3:])
        if kinds:
            self.last_failure_kind = kinds[-1]


def estimate_schema_tokens(ctx: ContextManager, tool_schemas: list[dict]) -> int:
    return ctx.estimate_tokens(json.dumps(tool_schemas, ensure_ascii=False))


def estimate_skill_tokens(ctx: ContextManager, skill_prompt: str) -> int:
    if not skill_prompt:
        return 0
    return ctx.estimate_tokens(skill_prompt) + 4


def render_token_report(
    ctx: ContextManager,
    tool_schemas: list[dict],
    skill_prompt: str,
    verbose: bool = False,
) -> str:
    schema_tokens = estimate_schema_tokens(ctx, tool_schemas)
    skill_tokens = estimate_skill_tokens(ctx, skill_prompt)
    diag = ctx.get_token_diagnostics(
        schema_tokens_estimate=schema_tokens,
        skill_tokens_estimate=skill_tokens,
    )
    utilization = diag["effective"] / ctx.max_tokens * 100

    if not verbose:
        lines = [
            f"Messages:     {len(ctx.messages)} ({len(ctx.messages) - 1} excluding system)",
            f"Requests:     {ctx.stats.total_requests}",
            "",
            "Cumulative API-reported usage:",
            f"  Prompt:     {ctx.stats.total_prompt_tokens:,} tokens",
            f"  Completion: {ctx.stats.total_completion_tokens:,} tokens",
            f"  Total:      {ctx.stats.total_tokens:,} tokens",
            "",
            "Context accounting:",
            f"  Managed:    ~{diag['structured']:,} tokens (messages we manage)",
            f"  Overhead:   ~{diag['overhead_estimate']:,} tokens (schemas + active skills prompt)",
            f"  Effective:  ~{diag['effective']:,} tokens (API-calibrated for budgeting)",
            f"  Limit:      {ctx.max_tokens:,} tokens",
            f"  Usage:      {utilization:.1f}%",
        ]
        return "\n".join(lines)

    lines = [
        f"Messages:     {len(ctx.messages)} ({len(ctx.messages) - 1} excluding system)",
        f"Requests:     {ctx.stats.total_requests}",
        "",
        "Cumulative API-reported usage:",
        f"  Prompt:     {ctx.stats.total_prompt_tokens:,} tokens",
        f"  Completion: {ctx.stats.total_completion_tokens:,} tokens",
        f"  Total:      {ctx.stats.total_tokens:,} tokens",
        "",
        "Current context (verbose):",
        f"  Managed:    ~{diag['structured']:,} tokens (local messages + metadata)",
        f"  Content:    ~{diag['content_only']:,} tokens (local content-only)",
        f"  Schemas:    ~{diag['schema_estimate']:,} tokens (active tool schemas)",
        f"  Skills:     ~{diag['skill_estimate']:,} tokens (active skill prompt)",
        f"  Overhead:   ~{diag['overhead_estimate']:,} tokens (schemas + skills prompt)",
        f"  Effective:  ~{diag['effective']:,} tokens (API-calibrated, used for budget)",
        f"  Hidden*:    ~{diag['hidden_overhead_estimate']:,} tokens",
        f"  Limit:      {ctx.max_tokens:,} tokens",
        f"  Usage:      {utilization:.1f}%",
        "",
        "* Hidden ~= provider formatting + tokenizer mismatch + system-internal overhead.",
    ]
    return "\n".join(lines)


def do_compact(
    client: OpenAI,
    model: str,
    ctx: ContextManager,
    current_overhead_tokens: int = 0,
    level: str = "low",
    compact_client: OpenAI | None = None,
    compact_model: str | None = None,
) -> None:
    """Compress old conversation messages.

    Args:
        level: "low" (conservative) or "high" (aggressive).
        compact_client: separate LLM client for compaction; falls back to *client*.
        compact_model: model name for compaction; falls back to *model*.
    """
    keep_recent = 6
    droppable = len(ctx.messages) - 1 - keep_recent
    if droppable < 4:
        log(
            f"  Not enough old messages to compact "
            f"({len(ctx.messages) - 1} total, need >{keep_recent + 3}).\n"
        )
        return

    _client = compact_client if compact_client is not None else client
    _model = compact_model if compact_model else model

    before = ctx.get_context_tokens(overhead_tokens=current_overhead_tokens)
    before_local = ctx.estimate_messages_tokens_structured()
    old_messages = ctx.messages[1:-keep_recent]
    log(f"  Compacting {len(old_messages)} old messages (level={level}, model={_model})...")

    compacted = compact_messages(_client, _model, old_messages, level=level)
    if not compacted:
        log("  Compaction failed - context unchanged.\n")
        return

    candidate_local = ctx.estimate_messages_tokens_structured(
        [ctx.messages[0]] + compacted + ctx.messages[-keep_recent:]
    )
    if candidate_local >= before_local:
        log("  Compacted version is not smaller - skipped.\n")
        return

    replaced, new_count = ctx.apply_compacted_messages(compacted, keep_recent)
    after = ctx.get_context_tokens(overhead_tokens=current_overhead_tokens)
    log(
        f"  Compressed {replaced} old messages → {new_count} summary message(s) "
        f"(+ {keep_recent} recent kept). "
        f"Context: ~{before:,} → ~{after:,} tokens\n"
    )


def handle_slash_command(
    command: str,
    client: OpenAI,
    model: str,
    ctx: ContextManager,
    runtime_state: dict,
) -> bool:
    cmd = command.strip().lower()
    if cmd == "/help":
        log(f"\n{SLASH_COMMANDS_HELP}\n")
        return True
    if cmd == "/task":
        task_dir = runtime_state.get("task_dir")
        task_src = runtime_state.get("task_context_source")
        log("\nTask context:")
        log(f"  task_dir: {task_dir if task_dir else '(none loaded)'}")
        log(f"  workspace: {workspace_root_str()}")
        log(f"  task_md: {task_src if task_src else '(none)'}")
        log("  tip: /task load <specifier>  |  /task inject\n")
        return True
    if cmd.startswith("/task load"):
        spec = command.strip()[len("/task load"):].strip()
        if not spec:
            log("\nUsage: /task load <specifier>\n")
            return True
        try:
            task_dir = resolve_task_path(spec)
            workdir = task_dir / "workdir"
            workspace = setup_workspace(task_dir, workdir)
            set_workspace_root(workspace)
            history_prompt = load_history_prompt(task_dir)
            task_prompt, task_src = load_task_prompt(task_dir)
        except Exception as e:
            log(f"\nFailed to load task: {e}\n")
            return True

        runtime_state["task_dir"] = task_dir
        runtime_state["history_prompt"] = history_prompt
        runtime_state["task_prompt"] = task_prompt
        runtime_state["task_context_source"] = str(task_src) if task_src else ""

        ctx.system_prompt = compose_system_prompt(history_prompt, task_prompt)
        ctx.clear()
        log("\nTask loaded:")
        log(f"  task_dir: {task_dir}")
        log(f"  workspace: {workspace}")
        log(f"  history: {'loaded' if history_prompt else 'none'}")
        log(f"  task_md: {task_src if task_src else 'none'}")
        log(f"\n{workspace_summary()}\n")
        return True
    if cmd == "/task reload":
        task_dir = runtime_state.get("task_dir")
        if not task_dir:
            log("\nNo active task. Use /task load <specifier> first.\n")
            return True
        try:
            history_prompt = load_history_prompt(task_dir)
            task_prompt, task_src = load_task_prompt(task_dir)
        except Exception as e:
            log(f"\nFailed to reload task context: {e}\n")
            return True
        runtime_state["history_prompt"] = history_prompt
        runtime_state["task_prompt"] = task_prompt
        runtime_state["task_context_source"] = str(task_src) if task_src else ""
        ctx.system_prompt = compose_system_prompt(history_prompt, task_prompt)
        ctx.clear()
        log("\nTask context reloaded and conversation reset.")
        log(f"  history: {'loaded' if history_prompt else 'none'}")
        log(f"  task_md: {task_src if task_src else 'none'}\n")
        return True
    if cmd == "/task inject":
        task_dir = runtime_state.get("task_dir")
        if not task_dir:
            log("\nNo active task. Use /task load <specifier> first.\n")
            return True
        task_prompt, task_src = load_task_prompt(task_dir)
        if not task_prompt:
            log("\nNo task.md/TASK.md found (or file is empty).\n")
            return True
        runtime_state["task_prompt"] = task_prompt
        runtime_state["task_context_source"] = str(task_src) if task_src else ""
        ctx.messages.append(
            {
                "role": "system",
                "content": (
                    "## Runtime Task Context Injection\n"
                    "Use this newly injected task context for subsequent turns.\n\n"
                    + task_prompt
                ),
            }
        )
        log(f"\nInjected task context from: {task_src}\n")
        return True
    if cmd == "/tokens":
        log(
            "\n"
            + render_token_report(
                ctx,
                runtime_state["active_tool_schemas"],
                runtime_state["active_skill_prompt"],
                verbose=runtime_state["verbose"],
            )
            + "\n"
        )
        return True
    if cmd == "/history":
        log(f"\n{ctx.format_history()}\n")
        return True
    if cmd in ("/debug", "/debug raw"):
        if cmd == "/debug raw":
            log(f"\n{ctx.format_raw()}\n")
        else:
            log(f"\n{ctx.format_debug()}\n")
        return True
    if cmd.startswith("/compact"):
        parts = command.strip().split()
        level = "low"
        if len(parts) >= 2:
            arg = parts[1].lower()
            if arg in ("low", "high"):
                level = arg
            else:
                log(f"\nUsage: /compact [low|high]  (default: low)\n")
                return True
        overhead_tokens = estimate_schema_tokens(
            ctx, runtime_state["active_tool_schemas"]
        ) + estimate_skill_tokens(ctx, runtime_state["active_skill_prompt"])
        do_compact(
            client,
            model,
            ctx,
            current_overhead_tokens=overhead_tokens,
            level=level,
            compact_client=runtime_state.get("compact_client"),
            compact_model=runtime_state.get("compact_model"),
        )
        return True
    if cmd == "/skills":
        all_skill_names = sorted(runtime_state["skills"].keys())
        active = runtime_state.get("active_skill_names", [])
        pinned = sorted(runtime_state["pinned_skills"])
        log("\nSkills:")
        log(f"  Loaded:  {', '.join(all_skill_names) if all_skill_names else '(none)'}")
        log(f"  Active:  {', '.join(active) if active else '(none)'}")
        log(f"  Pinned:  {', '.join(pinned) if pinned else '(none)'}\n")
        return True
    if cmd.startswith("/skill "):
        parts = cmd.split()
        if len(parts) != 3:
            log("\nUsage: /skill <name> on|off\n")
            return True
        _, name, mode = parts
        if name not in runtime_state["skills"]:
            log(f"\nUnknown skill: {name}\n")
            return True
        if mode == "on":
            runtime_state["pinned_skills"].add(name)
            log(f"\nPinned skill: {name}\n")
        elif mode == "off":
            runtime_state["pinned_skills"].discard(name)
            log(f"\nUnpinned skill: {name}\n")
        else:
            log("\nUsage: /skill <name> on|off\n")
        return True
    if cmd.startswith("/verbose"):
        parts = cmd.split()
        if len(parts) == 1:
            state = "on" if runtime_state["verbose"] else "off"
            log(f"\nVerbose token diagnostics: {state}\n")
            return True
        arg = parts[1]
        if arg in ("on", "true", "1"):
            runtime_state["verbose"] = True
            log("\nVerbose token diagnostics: on\n")
            return True
        if arg in ("off", "false", "0"):
            runtime_state["verbose"] = False
            log("\nVerbose token diagnostics: off\n")
            return True
        log("\nUsage: /verbose on|off\n")
        return True
    if cmd.startswith("/shell-safe"):
        parts = cmd.split()
        if len(parts) == 1:
            state = "on" if runtime_state["safe_shell"] else "off"
            log(f"\nSafe shell mode: {state}\n")
            return True
        arg = parts[1]
        if arg in ("on", "true", "1"):
            runtime_state["safe_shell"] = True
            set_shell_safety(True)
            log("\nSafe shell mode: on\n")
            return True
        if arg in ("off", "false", "0"):
            runtime_state["safe_shell"] = False
            set_shell_safety(False)
            log("\nSafe shell mode: off\n")
            return True
        log("\nUsage: /shell-safe on|off\n")
        return True
    if cmd == "/shell-policy":
        snap = get_shell_policy_snapshot()
        log("\nShell policy:")
        log(f"  safe_mode: {snap['safe_mode']}")
        log(f"  policy_file: {snap['policy_file']}")
        log(f"  allowlist: {len(snap['allowlist'])} entries")
        log(f"  denylist: {len(snap['denylist'])} entries")
        if snap["allowlist"]:
            log("  allowlist entries:")
            for x in snap["allowlist"]:
                log(f"    - {x}")
        if snap["denylist"]:
            log("  denylist entries:")
            for x in snap["denylist"]:
                log(f"    - {x}")
        log()
        return True
    if cmd == "/preempt":
        mode = "on" if runtime_state.get("preempt_shell_kill", False) else "off"
        log("\nPreemption settings:")
        log("  queue: enabled")
        log("  soft_preempt: enabled")
        log(f"  shell_kill_on_preempt: {mode}\n")
        return True
    if cmd.startswith("/preempt shell-kill"):
        parts = cmd.split()
        if len(parts) != 3:
            log("\nUsage: /preempt shell-kill on|off\n")
            return True
        arg = parts[2]
        if arg in ("on", "true", "1"):
            runtime_state["preempt_shell_kill"] = True
            set_shell_interrupt_on_preempt(True)
            log("\nPreemption shell-kill: on\n")
            return True
        if arg in ("off", "false", "0"):
            runtime_state["preempt_shell_kill"] = False
            set_shell_interrupt_on_preempt(False)
            log("\nPreemption shell-kill: off\n")
            return True
        log("\nUsage: /preempt shell-kill on|off\n")
        return True
    if cmd.startswith("/recovery"):
        parts = cmd.split()
        if len(parts) == 1:
            state = "on" if runtime_state.get("recovery_cleanup", True) else "off"
            log("\nRecovery cleanup:")
            log(f"  remove failed intermediate traces after success: {state}\n")
            return True
        arg = parts[1]
        if arg in ("on", "true", "1"):
            runtime_state["recovery_cleanup"] = True
            log("\nRecovery cleanup: on\n")
            return True
        if arg in ("off", "false", "0"):
            runtime_state["recovery_cleanup"] = False
            log("\nRecovery cleanup: off\n")
            return True
        log("\nUsage: /recovery on|off\n")
        return True
    if cmd == "/workspace":
        log(f"\n{workspace_summary()}\n")
        return True
    if cmd == "/clear":
        ctx.clear()
        log("\nConversation cleared.\n")
        return True
    return False


def format_turn_usage(usage) -> str:
    prompt = getattr(usage, "prompt_tokens", 0) or 0
    completion = getattr(usage, "completion_tokens", 0) or 0
    total = getattr(usage, "total_tokens", 0) or 0
    return f"[tokens: prompt={prompt:,}, completion={completion:,}, total={total:,}]"


def is_procedural_confirmation(text: str) -> bool:
    low = text.lower()
    cues = (
        "proceed?", "reply `yes` or `no`", "reply yes or no",
        "wait for your go-ahead", "action: proceed", "approve",
        "approval", "go-ahead", "allow once", "always allow",
    )
    return any(c in low for c in cues)


def unique_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def resolve_active_tool_schemas(
    selected_skills,
    tool_schema_map: dict[str, dict],
    all_tool_schemas: list[dict],
) -> list[dict]:
    selected_tool_names = [
        tool
        for skill in selected_skills
        for tool in skill.tools
        if tool in tool_schema_map
    ]
    if not selected_tool_names:
        return all_tool_schemas
    return [tool_schema_map[name] for name in unique_preserve_order(selected_tool_names)]


def maybe_extend_skills_for_continuation(
    user_input: str,
    selected_skills,
    runtime_state: dict,
) -> None:
    if user_input.lower() not in CONTINUATION_REPLIES:
        return
    existing = {s.name for s in selected_skills}
    for prev_name in runtime_state.get("active_skill_names", []):
        if prev_name in runtime_state["skills"] and prev_name not in existing:
            selected_skills.append(runtime_state["skills"][prev_name])


def has_action_intent(user_input: str, selected_skill_names: list[str]) -> bool:
    _ = selected_skill_names  # preserved for compatibility at callsites
    text = user_input.strip().lower()
    if not text:
        return False
    if text.startswith("/"):
        return False
    # Action intent should be inferred from user utterance itself, not always-on skills.
    return bool(ACTION_INTENT_RE.search(text))


def is_tool_failure(tool_name: str, result: str) -> bool:
    text = (result or "").strip()
    if not text:
        return False
    if text.startswith("Error") or text.startswith("Denied"):
        return True
    if tool_name == "run_shell":
        match = EXIT_CODE_RE.search(text)
        if match:
            try:
                return int(match.group(1)) != 0
            except ValueError:
                return True
    return False


def summarize_failure(tool_name: str, result: str) -> str:
    first_line = (result or "").strip().splitlines()[0] if result else ""
    if not first_line:
        first_line = "unknown failure"
    return f"{tool_name}: {first_line}"


def build_recovery_nudge(recovery: RecoveryState) -> str:
    """Build a CUDA-aware recovery nudge based on failure kind."""
    latest = recovery.failures[-1] if recovery.failures else "unknown failure"
    repeat_hint = ""
    if recovery.repeated_failure_count >= 2:
        repeat_hint = (
            "You repeated a similar failing attempt. Change strategy and do not retry the same "
            "command/arguments unchanged.\n"
        )

    kind = recovery.last_failure_kind
    if kind == CudaFailureKind.COMPILE:
        specific = (
            "This is a COMPILATION error. Common causes:\n"
            "- Missing extern \"C\" declarations or mismatched launcher signatures\n"
            "- Incorrect #include paths (use <torch/types.h> not <torch/extension.h>)\n"
            "- CUDA syntax errors in .cu files\n"
            "- Missing REGISTER_BINDING macro\n"
            "Read the compiler error output carefully and fix the specific issue.\n"
        )
    elif kind == CudaFailureKind.CORRECTNESS:
        specific = (
            "This is a CORRECTNESS error. Common causes:\n"
            "- Boundary conditions: missing tid < size check in kernel\n"
            "- Wrong math or indexing logic in the CUDA kernel\n"
            "- Missing __syncthreads() before shared memory reuse\n"
            "- Data type mismatch (e.g. int vs float)\n"
            "- model_new.py not matching Model's interface exactly\n"
            "Debug by comparing kernel logic with the original PyTorch forward pass.\n"
        )
    elif kind == CudaFailureKind.PERFORMANCE:
        specific = (
            "Correctness passed but PERFORMANCE is insufficient (need >= 5% faster than torch.compile).\n"
            "Apply optimizations in priority order:\n"
            "1. Kernel fusion to reduce memory traffic\n"
            "2. Shared memory tiling for data reuse\n"
            "3. Vectorized loads (float2/float4)\n"
            "4. Occupancy tuning (adjust block size)\n"
        )
    else:
        specific = ""

    return (
        "Recovery mode: latest tool execution failed.\n"
        f"- Latest failure: {latest}\n"
        f"{specific}"
        "- Diagnose root cause from tool output and perform a concrete fix in the next tool call.\n"
        "- Do not ask user for confirmation; continue autonomously.\n"
        f"{repeat_hint}"
    ).strip()


def process_tool_calls(response, ctx: ContextManager) -> ToolExecutionOutcome:
    message = response.choices[0].message
    if not message.tool_calls:
        return ToolExecutionOutcome(called=False, failures=[], failure_kinds=[])

    ctx.add_assistant_tool_calls(message)
    failures: list[str] = []
    failure_kinds: list[str] = []
    for tool_call in message.tool_calls:
        name = (tool_call.function.name or "").strip()
        args = tool_call.function.arguments
        if not name:
            result = "Error: empty tool name returned by model."
            log(f"  <- Result: {result}")
            ctx.add_tool_result(tool_call.id, "unknown", result)
            failures.append("unknown: empty tool name")
            failure_kinds.append(CudaFailureKind.GENERAL)
            continue
        log(f"  -> Calling tool: {name}({args})")
        result = execute_tool(name, args)
        display = result if len(result) <= 2000 else result[:1000] + "\n...(truncated)...\n" + result[-500:]
        log(f"  <- Result: {display}")
        ctx.add_tool_result(tool_call.id, name, result)
        if is_tool_failure(name, result):
            failures.append(summarize_failure(name, result))
            failure_kinds.append(classify_cuda_failure(name, result))
    return ToolExecutionOutcome(called=True, failures=failures, failure_kinds=failure_kinds)


def _extract_profile_from_context(ctx: ContextManager) -> dict:
    """Scan recent context messages for profiling results."""
    for msg in reversed(ctx.messages[-10:]):
        content = msg.get("content", "") or ""
        m = PROFILE_RESULT_RE.search(content)
        if m:
            try:
                return {
                    "baseline_us": float(m.group(1)),
                    "compile_us": float(m.group(2)),
                    "cuda_us": float(m.group(3)),
                }
            except ValueError:
                pass
    return {}


def _try_save_history(task_dir: Path, assistant_content: str, ctx: ContextManager | None = None) -> None:
    """Save workdir artifacts to history if verification passed."""
    from cuda_task import get_workspace_path
    workdir = get_workspace_path()
    if not workdir or not (workdir / "model_new.py").is_file():
        return
    profile_result: dict = {}
    m = PROFILE_RESULT_RE.search(assistant_content)
    if m:
        try:
            profile_result = {
                "baseline_us": float(m.group(1)),
                "compile_us": float(m.group(2)),
                "cuda_us": float(m.group(3)),
            }
        except ValueError:
            pass
    if not profile_result and ctx:
        profile_result = _extract_profile_from_context(ctx)
    try:
        save_to_history(task_dir, workdir, profile_result)
        log(f"  [history] Saved successful implementation to {task_dir / 'history'}")
    except Exception as e:
        log(f"  [history] Failed to save: {e}")


def chat(
    client: OpenAI,
    model: str,
    max_tokens: int,
    max_agent_steps: int,
    safe_shell: bool,
    recovery_cleanup: bool,
    preempt_shell_kill: bool,
    initial_message: str | None = None,
    task_dir: Path | None = None,
    history_prompt: str = "",
    task_prompt: str = "",
    task_context_source: str = "",
    compact_client: OpenAI | None = None,
    compact_model: str | None = None,
) -> None:
    set_shell_safety(safe_shell)
    set_shell_interrupt_on_preempt(preempt_shell_kill)

    system_prompt = compose_system_prompt(history_prompt, task_prompt)
    ctx = ContextManager(system_prompt=system_prompt, max_tokens=max_tokens)
    all_tool_schemas = get_all_tool_schemas()
    tool_schema_map = get_tool_schema_map()
    skills = load_skills(Path(__file__).resolve().parent / "skills")

    default_selected_skills = select_skills("", skills, pinned_on=set())
    default_tool_names = [
        tool
        for skill in default_selected_skills
        for tool in skill.tools
        if tool in tool_schema_map
    ]
    if not default_tool_names:
        default_tool_schemas = all_tool_schemas
        default_skill_prompt = ""
        default_skill_names = []
    else:
        default_tool_schemas = [
            tool_schema_map[name] for name in unique_preserve_order(default_tool_names)
        ]
        default_skill_prompt = build_skill_prompt(default_selected_skills)
        default_skill_names = [s.name for s in default_selected_skills]

    runtime_state = {
        "verbose": False,
        "safe_shell": safe_shell,
        "recovery_cleanup": recovery_cleanup,
        "skills": skills,
        "pinned_skills": set(),
        "active_skill_names": default_skill_names,
        "active_tool_schemas": default_tool_schemas,
        "active_skill_prompt": default_skill_prompt,
        "task_dir": task_dir,
        "history_prompt": history_prompt,
        "task_prompt": task_prompt,
        "task_context_source": task_context_source,
        "preempt_shell_kill": preempt_shell_kill,
        "compact_client": compact_client,
        "compact_model": compact_model,
    }

    log(f"CUDA Agent ready (model: {model}, context: {max_tokens:,} tokens).")
    log(f"Tools loaded: {', '.join(t['function']['name'] for t in all_tool_schemas)}")
    if skills:
        log(f"Skills loaded: {', '.join(sorted(skills.keys()))}")
    log(f"Safe shell mode: {'on' if safe_shell else 'off'}")
    log(f"Preempt shell-kill: {'on' if preempt_shell_kill else 'off'}")
    log(f"Recovery cleanup: {'on' if recovery_cleanup else 'off'}")
    log(f"Max autonomous steps per turn: {max_agent_steps}")
    _cm = compact_model or model
    log(f"Compact model: {_cm}")
    log(f"Workspace: {workspace_root_str()}")
    if history_prompt:
        log(f"History: loaded previous implementation for reference")
    if task_context_source:
        log(f"Task context: loaded from {task_context_source}")
    log("Type /help for commands, 'quit' to exit.\n")

    user_queue: Queue[tuple[str, str]] = Queue()
    reader_stop_event = threading.Event()
    reader_thread = InputReaderThread(user_queue, reader_stop_event)
    reader_thread.start()
    pending_input = initial_message

    try:
        while True:
            if pending_input is not None:
                user_input = pending_input
                pending_input = None
            else:
                log(">>> Ready for input.")
                try:
                    event, text = _dequeue_user_input_blocking(user_queue)
                except KeyboardInterrupt:
                    log("\nBye!")
                    break
                if event in (QUEUE_EVENT_EOF, QUEUE_EVENT_INTERRUPT):
                    log("\nBye!")
                    break
                user_input = text

            if not user_input:
                continue
            log(f"You: {user_input}")
            if user_input.lower() in ("quit", "exit", "q"):
                log("Bye!")
                break
            if user_input.startswith("/"):
                handle_slash_command(user_input, client, model, ctx, runtime_state)
                continue

            turn_start_index = len(ctx.messages)
            ctx.add_user_message(user_input)

            selected_skills = select_skills(
                user_input, runtime_state["skills"], runtime_state["pinned_skills"]
            )
            maybe_extend_skills_for_continuation(user_input, selected_skills, runtime_state)
            selected_skill_names = [s.name for s in selected_skills]
            active_tool_schemas = resolve_active_tool_schemas(
                selected_skills, tool_schema_map, all_tool_schemas
            )

            active_skill_prompt = build_skill_prompt(selected_skills)
            runtime_state["active_skill_names"] = selected_skill_names
            runtime_state["active_tool_schemas"] = active_tool_schemas
            runtime_state["active_skill_prompt"] = active_skill_prompt

            if selected_skill_names:
                log(f"  [skills] active: {', '.join(selected_skill_names)}")

            overhead_tokens = estimate_schema_tokens(
                ctx, active_tool_schemas
            ) + estimate_skill_tokens(ctx, active_skill_prompt)
            if ctx.needs_compaction(overhead_tokens=overhead_tokens):
                log("  (Context approaching limit, auto-compacting...)")
                do_compact(
                    client,
                    model,
                    ctx,
                    current_overhead_tokens=overhead_tokens,
                    level="low",
                    compact_client=runtime_state.get("compact_client"),
                    compact_model=runtime_state.get("compact_model"),
                )

            likely_action_intent = has_action_intent(user_input, selected_skill_names)
            autonomy_nudge = ""
            did_call_tool = False
            recovery = RecoveryState()
            preempted = False
            # Deduplicate history saves within a single turn: key = (baseline, compile, cuda)
            saved_profile_sigs: set[tuple] = set()
            set_autonomous_turn(True)

            for step_idx in range(max_agent_steps):
                try:
                    pending_from_preempt = _to_pending_input_from_preempt_event(
                        _dequeue_user_input_nowait(user_queue), ctx
                    )
                    if pending_from_preempt:
                        pending_input = pending_from_preempt
                        preempted = True
                        clear_preempt_request()
                        if pending_input == "quit":
                            log("  [preempt] Received terminal interrupt signal; finishing current turn.")
                        else:
                            log("  [preempt] Received user input during autonomous loop; switching turns.")
                        break

                    request_messages = list(ctx.messages)
                    if autonomy_nudge:
                        request_messages.append({"role": "system", "content": autonomy_nudge})
                    if active_skill_prompt:
                        request_messages.append(
                            {"role": "system", "content": active_skill_prompt}
                        )
                    # Make model request asynchronously so preempt can be observed while waiting.
                    async_call = _start_async_model_call(
                        client=client,
                        model=model,
                        request_messages=request_messages,
                        active_tool_schemas=active_tool_schemas,
                    )
                    preempted_while_waiting = False
                    while not async_call.done.wait(0.1):
                        pending_from_preempt = _to_pending_input_from_preempt_event(
                            _dequeue_user_input_nowait(user_queue), ctx
                        )
                        if pending_from_preempt:
                            pending_input = pending_from_preempt
                            preempted = True
                            preempted_while_waiting = True
                            clear_preempt_request()
                            if pending_input == "quit":
                                log("  [preempt] Received terminal interrupt while waiting model response.")
                            else:
                                log("  [preempt] Preempted in-flight model request; switching turns.")
                            break
                    if preempted_while_waiting:
                        break
                    if async_call.error is not None:
                        raise async_call.error
                    if async_call.response is None:
                        raise RuntimeError("Model call ended without response.")
                    response = async_call.response
                    ctx.record_usage(response.usage, overhead_tokens=overhead_tokens)

                    tool_outcome = process_tool_calls(response, ctx)
                    if tool_outcome.called:
                        did_call_tool = True
                        # Save history immediately when profiling results appear in any tool
                        # output — ensures every intermediate good result is preserved even
                        # if the agent continues optimising and later overwrites the workdir.
                        if task_dir:
                            for msg in reversed(ctx.messages[-6:]):
                                if msg.get("role") != "tool":
                                    continue
                                m_p = PROFILE_RESULT_RE.search(msg.get("content", "") or "")
                                if m_p:
                                    try:
                                        sig = (m_p.group(1), m_p.group(2), m_p.group(3))
                                        if sig not in saved_profile_sigs:
                                            saved_profile_sigs.add(sig)
                                            _try_save_history(task_dir, msg.get("content", ""))
                                    except (ValueError, IndexError):
                                        pass
                                    break
                        if tool_outcome.failures:
                            recovery.record_failures(tool_outcome.failures, tool_outcome.failure_kinds)
                            log(f"  [recovery] Detected failure ({recovery.last_failure_kind}): "
                                f"{tool_outcome.failures[-1]}")
                            autonomy_nudge = build_recovery_nudge(recovery)
                        else:
                            recovery.unresolved_failure = False
                            autonomy_nudge = ""
                        continue

                    content = response.choices[0].message.content or ""
                    if (
                        likely_action_intent
                        and (
                            is_procedural_confirmation(content)
                            or recovery.unresolved_failure
                        )
                        and step_idx < max_agent_steps - 1
                    ):
                        if recovery.unresolved_failure:
                            log("  [recovery] Suppressed unresolved-failure reply; continuing...")
                            autonomy_nudge = build_recovery_nudge(recovery)
                        else:
                            log("  [autonomy] Suppressed procedural-confirmation reply; continuing...")
                            autonomy_nudge = AUTONOMY_NUDGE_TEXT
                        continue

                    ctx.add_assistant_message(content)
                    log(f"\nAssistant: {content}")
                    log(f"  {format_turn_usage(response.usage)}\n")
                    if runtime_state.get("recovery_cleanup", True) and recovery.had_failure:
                        removed = ctx.drop_failed_tool_messages(start_index=turn_start_index + 1)
                        if removed > 0:
                            log(f"  [recovery] Cleaned {removed} failed intermediate messages from context.\n")

                    # Save successful result to history if task_dir is set
                    if task_dir:
                        _try_save_history(task_dir, content, ctx)

                    break
                except Exception as e:
                    log(f"\nError: {e}\n")
                    ctx.pop_last_message()
                    break
            else:
                help_msg = (
                    f"I reached the max autonomous step limit ({max_agent_steps}) for this turn "
                    "and may be stuck. Please provide guidance, constraints, or the next action."
                )
                if recovery.failures:
                    help_msg += f" Latest failure: {recovery.failures[-1]}"
                ctx.add_assistant_message(help_msg)
                log(f"\nAssistant: {help_msg}\n")

            set_autonomous_turn(False)
            clear_preempt_request()
            if preempted:
                continue
    finally:
        set_autonomous_turn(False)
        clear_preempt_request()
        reader_stop_event.set()
        reader_thread.join(timeout=1.0)


def main():
    parser = argparse.ArgumentParser(
        description="Preemptible CUDA Kernel Development Agent"
    )
    parser.add_argument("--list-models", action="store_true", help="List models and exit")
    parser.add_argument(
        "--model", type=str, default=None, help="Model to use (default: provider-specific)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Context window size (default: {DEFAULT_MAX_TOKENS:,})",
    )
    parser.add_argument(
        "--max-agent-steps",
        type=int,
        default=DEFAULT_MAX_AGENT_STEPS,
        help=f"Max autonomous LLM/tool rounds per user turn (default: {DEFAULT_MAX_AGENT_STEPS})",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task specifier: directory path (task/example_axpby), "
             "dataset id (level1/3), or global numeric id (42).",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List available dataset tasks and exit.",
    )
    parser.add_argument(
        "--workdir",
        type=str,
        default=None,
        help="Working directory for the task (default: <task_dir>/workdir)",
    )
    parser.add_argument(
        "--safe-shell",
        action="store_true",
        help="Enable safe shell mode (default: off for CUDA agent).",
    )
    parser.add_argument(
        "--keep-recovery-trace",
        action="store_true",
        help="Keep failed intermediate traces in context after successful task completion.",
    )
    parser.add_argument(
        "--preempt-shell-kill",
        action="store_true",
        help="When new user input arrives during autonomous mode, terminate running shell commands.",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="auto",
        help="GPU selection: 'auto' (default) auto-detects idle GPU via rocm-smi/nvidia-smi; "
             "'none' disables GPU env injection; or an explicit index like '1'.",
    )
    parser.add_argument(
        "--compact-model",
        type=str,
        default=None,
        help="Model to use for /compact (default: deepseek-v3). Uses the same API client.",
    )
    args = parser.parse_args()

    if args.list_tasks:
        print(list_tasks())
        return

    client, provider_model = create_client()
    model = args.model or provider_model
    if args.list_models:
        list_models(client)
        return

    # Compact model: default gpt-oss-120b (cheap on Ksyun), configurable via --compact-model.
    # Falls back to provider_model if not overridden.
    compact_model: str | None = args.compact_model or "gpt-oss-120b"

    gpu_flag = (args.gpu or "").strip().lower()
    if gpu_flag == "auto":
        set_gpu_auto(True)
    elif gpu_flag and gpu_flag != "none":
        set_gpu_device(args.gpu.strip())

    # Resolve optional task specifier
    task_dir: Path | None = None
    if args.task:
        try:
            task_dir = resolve_task_path(args.task)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error: {e}")
            sys.exit(1)

    # Set up run logger
    global _logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = (task_dir / "logs") if task_dir else (Path(__file__).resolve().parent / "logs")
    log_path = log_dir / f"{timestamp}.log"
    _logger = RunLogger(log_path)

    history_prompt = ""
    task_prompt = ""
    task_context_source = ""
    initial_message = None

    if task_dir:
        workdir = Path(args.workdir).resolve() if args.workdir else task_dir / "workdir"
        log(f"Setting up workspace: task={task_dir}, workdir={workdir}, gpu={args.gpu}")
        workspace = setup_workspace(task_dir, workdir)
        set_workspace_root(workspace)
        log(f"Workspace ready:\n{workspace_summary()}")
        if gpu_flag == "auto":
            log(f"GPU status: {gpu_status_summary()}")

        # Load optional context for this task
        history_prompt = load_history_prompt(task_dir)
        if history_prompt:
            log(f"Loaded history from {task_dir / 'history'}")
        task_prompt, task_src = load_task_prompt(task_dir)
        if task_src:
            task_context_source = str(task_src)
            log(f"Loaded task context from {task_src}")
        initial_message = "Optimize the PyTorch model in model.py by implementing custom CUDA kernels."
    else:
        # Chat-first mode: user can inject task later via /task load <specifier>
        set_workspace_root(Path(__file__).resolve().parent)
        log("No task specified. Chat mode enabled.")
        log("Use /task load <specifier> to load a task and initialize workdir.")
    log()

    try:
        chat(
            client=client,
            model=model,
            max_tokens=args.max_tokens,
            max_agent_steps=max(1, args.max_agent_steps),
            safe_shell=args.safe_shell,
            recovery_cleanup=not args.keep_recovery_trace,
            preempt_shell_kill=args.preempt_shell_kill,
            initial_message=initial_message,
            task_dir=task_dir,
            history_prompt=history_prompt,
            task_prompt=task_prompt,
            task_context_source=task_context_source,
            compact_client=client,
            compact_model=compact_model,
        )
    finally:
        log(f"\nLog saved to: {log_path}")
        _logger.close()


if __name__ == "__main__":
    main()
