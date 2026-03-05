"""
07_cuda_agent - Autonomous CUDA kernel development agent.

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
"""

import argparse
from dataclasses import dataclass, field
from datetime import datetime
import json
import re
import sys
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm import create_client, list_models  # noqa: E402

from compactor import compact_messages
from context import ContextManager
from cuda_task import (
    list_tasks,
    load_history_prompt,
    resolve_task_path,
    save_to_history,
    setup_workspace,
    workspace_summary,
)
from skill_manager import build_skill_prompt, load_skills, select_skills
from gpu_pool import gpu_status_summary
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
DEFAULT_MAX_AGENT_STEPS = 20


class RunLogger:
    """Tee-writes to stdout and a log file."""

    def __init__(self, log_path: Path | None = None):
        self._file = None
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(log_path, "w", encoding="utf-8")

    def log(self, msg: str = "") -> None:
        print(msg)
        if self._file:
            self._file.write(msg + "\n")
            self._file.flush()

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None


_logger = RunLogger()


def log(msg: str = "") -> None:
    _logger.log(msg)

_SYSTEM_PROMPT_TEMPLATE = """\
You are an expert CUDA kernel developer. Your task is to accelerate a PyTorch model \
by implementing custom CUDA C++ extensions that are both correct and fast.

Workspace root: {workspace_root}
All file-tool paths are relative to this root. run_shell executes in this directory by default.

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
8. Output a summary with profiling results and STOP.

## Stopping Rule
Once verification passes, run profiling ONCE for measurement, then output a final \
summary and STOP immediately. Do NOT continue optimizing. Do NOT write test scripts. \
Profiling is informational only — there is no performance gate.

## Tool Usage Policy
1. Use tools directly — do not ask for permission or provide a plan first.
2. For file tasks: read_file -> write_file. For shell tasks: run_shell.
3. Keep responses concise; focus on action.
4. When compilation or verification fails, diagnose from tool output and fix immediately.
5. Do NOT create helper/test scripts. Only use the standard compile/verify/profile pipeline.
"""


def get_system_prompt() -> str:
    return _SYSTEM_PROMPT_TEMPLATE.format(workspace_root=workspace_root_str())


SLASH_COMMANDS_HELP = """Available commands:
  /tokens       - Show token usage statistics
  /history      - Show all messages with token estimates (compact)
  /debug        - Show full context with colored output (what the LLM sees)
  /compact      - Smart context compaction (compress old messages)
  /skills       - Show loaded skills and currently active skills
  /skill        - Pin/unpin a skill: /skill <name> on|off
  /verbose      - Show or set verbose token diagnostics: /verbose on|off
  /shell-safe   - Toggle safe shell mode: /shell-safe on|off
  /shell-policy - Show current shell allowlist/denylist
  /recovery     - Show or set recovery cleanup: /recovery on|off
  /workspace    - Show current task workspace info
  /clear        - Reset conversation history
  /help         - Show this help message"""

CONTINUATION_REPLIES = {
    "y", "yes", "ok", "okay", "sure", "continue", "go", "proceed",
    "继续", "好的", "好",
}
ACTION_INTENT_RE = re.compile(
    r"\b(run|execute|check|list|read|write|search|find|show|compile|verify|profile|optimize)\b"
)
AUTONOMY_NUDGE_TEXT = (
    "Do not provide a plan or ask the user for procedural confirmation. "
    "For this turn, immediately call the required tools and only then provide a final summary."
)
EXIT_CODE_RE = re.compile(r"^exit_code=(-?\d+)$", re.MULTILINE)
HANDOFF_PHRASES = (
    "let me know",
    "if you'd like",
    "would you like",
    "please provide",
    "to proceed",
    "share the file",
    "share the code",
)


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
    client: OpenAI, model: str, ctx: ContextManager, current_overhead_tokens: int = 0
) -> None:
    keep_recent = 6
    droppable = len(ctx.messages) - 1 - keep_recent
    if droppable < 4:
        print(
            f"  Not enough old messages to compact "
            f"({len(ctx.messages) - 1} total, need >{keep_recent + 3}).\n"
        )
        return

    before = ctx.get_context_tokens(overhead_tokens=current_overhead_tokens)
    before_local = ctx.estimate_messages_tokens_structured()
    old_messages = ctx.messages[1:-keep_recent]
    print(f"  Compacting {len(old_messages)} old messages...")

    compacted = compact_messages(client, model, old_messages)
    if not compacted:
        print("  Compaction failed - context unchanged.\n")
        return

    candidate_local = ctx.estimate_messages_tokens_structured(
        [ctx.messages[0]] + compacted + ctx.messages[-keep_recent:]
    )
    if candidate_local >= before_local:
        print("  Compacted version is not smaller - skipped.\n")
        return

    replaced, new_count = ctx.apply_compacted_messages(compacted, keep_recent)
    after = ctx.get_context_tokens(overhead_tokens=current_overhead_tokens)
    print(
        f"  Replaced {replaced} messages with {new_count} compacted messages. "
        f"Context: ~{before:,} -> ~{after:,} tokens\n"
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
        print(f"\n{SLASH_COMMANDS_HELP}\n")
        return True
    if cmd == "/tokens":
        print(
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
        print(f"\n{ctx.format_history()}\n")
        return True
    if cmd == "/debug":
        print(f"\n{ctx.format_debug()}\n")
        return True
    if cmd == "/compact":
        overhead_tokens = estimate_schema_tokens(
            ctx, runtime_state["active_tool_schemas"]
        ) + estimate_skill_tokens(ctx, runtime_state["active_skill_prompt"])
        do_compact(client, model, ctx, current_overhead_tokens=overhead_tokens)
        return True
    if cmd == "/skills":
        all_skill_names = sorted(runtime_state["skills"].keys())
        active = runtime_state.get("active_skill_names", [])
        pinned = sorted(runtime_state["pinned_skills"])
        print("\nSkills:")
        print(f"  Loaded:  {', '.join(all_skill_names) if all_skill_names else '(none)'}")
        print(f"  Active:  {', '.join(active) if active else '(none)'}")
        print(f"  Pinned:  {', '.join(pinned) if pinned else '(none)'}\n")
        return True
    if cmd.startswith("/skill "):
        parts = cmd.split()
        if len(parts) != 3:
            print("\nUsage: /skill <name> on|off\n")
            return True
        _, name, mode = parts
        if name not in runtime_state["skills"]:
            print(f"\nUnknown skill: {name}\n")
            return True
        if mode == "on":
            runtime_state["pinned_skills"].add(name)
            print(f"\nPinned skill: {name}\n")
        elif mode == "off":
            runtime_state["pinned_skills"].discard(name)
            print(f"\nUnpinned skill: {name}\n")
        else:
            print("\nUsage: /skill <name> on|off\n")
        return True
    if cmd.startswith("/verbose"):
        parts = cmd.split()
        if len(parts) == 1:
            state = "on" if runtime_state["verbose"] else "off"
            print(f"\nVerbose token diagnostics: {state}\n")
            return True
        arg = parts[1]
        if arg in ("on", "true", "1"):
            runtime_state["verbose"] = True
            print("\nVerbose token diagnostics: on\n")
            return True
        if arg in ("off", "false", "0"):
            runtime_state["verbose"] = False
            print("\nVerbose token diagnostics: off\n")
            return True
        print("\nUsage: /verbose on|off\n")
        return True
    if cmd.startswith("/shell-safe"):
        parts = cmd.split()
        if len(parts) == 1:
            state = "on" if runtime_state["safe_shell"] else "off"
            print(f"\nSafe shell mode: {state}\n")
            return True
        arg = parts[1]
        if arg in ("on", "true", "1"):
            runtime_state["safe_shell"] = True
            set_shell_safety(True)
            print("\nSafe shell mode: on\n")
            return True
        if arg in ("off", "false", "0"):
            runtime_state["safe_shell"] = False
            set_shell_safety(False)
            print("\nSafe shell mode: off\n")
            return True
        print("\nUsage: /shell-safe on|off\n")
        return True
    if cmd == "/shell-policy":
        snap = get_shell_policy_snapshot()
        print("\nShell policy:")
        print(f"  safe_mode: {snap['safe_mode']}")
        print(f"  policy_file: {snap['policy_file']}")
        print(f"  allowlist: {len(snap['allowlist'])} entries")
        print(f"  denylist: {len(snap['denylist'])} entries")
        if snap["allowlist"]:
            print("  allowlist entries:")
            for x in snap["allowlist"]:
                print(f"    - {x}")
        if snap["denylist"]:
            print("  denylist entries:")
            for x in snap["denylist"]:
                print(f"    - {x}")
        print()
        return True
    if cmd.startswith("/recovery"):
        parts = cmd.split()
        if len(parts) == 1:
            state = "on" if runtime_state.get("recovery_cleanup", True) else "off"
            print("\nRecovery cleanup:")
            print(f"  remove failed intermediate traces after success: {state}\n")
            return True
        arg = parts[1]
        if arg in ("on", "true", "1"):
            runtime_state["recovery_cleanup"] = True
            print("\nRecovery cleanup: on\n")
            return True
        if arg in ("off", "false", "0"):
            runtime_state["recovery_cleanup"] = False
            print("\nRecovery cleanup: off\n")
            return True
        print("\nUsage: /recovery on|off\n")
        return True
    if cmd == "/workspace":
        print(f"\n{workspace_summary()}\n")
        return True
    if cmd == "/clear":
        ctx.clear()
        print("\nConversation cleared.\n")
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
    return (
        bool(ACTION_INTENT_RE.search(user_input.lower()))
        or ("shell" in selected_skill_names)
        or ("cuda" in selected_skill_names)
    )


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


def is_handoff_to_user_reply(text: str) -> bool:
    low = (text or "").lower()
    if any(p in low for p in HANDOFF_PHRASES):
        return True
    return low.strip().endswith("?") and ("you" in low or "your" in low)


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
    initial_message: str | None = None,
    task_dir: Path | None = None,
    history_prompt: str = "",
) -> None:
    set_shell_safety(safe_shell)

    system_prompt = get_system_prompt()
    if history_prompt:
        system_prompt += "\n\n" + history_prompt
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
    }

    log(f"CUDA Agent ready (model: {model}, context: {max_tokens:,} tokens).")
    log(f"Tools loaded: {', '.join(t['function']['name'] for t in all_tool_schemas)}")
    if skills:
        log(f"Skills loaded: {', '.join(sorted(skills.keys()))}")
    log(f"Safe shell mode: {'on' if safe_shell else 'off'}")
    log(f"Recovery cleanup: {'on' if recovery_cleanup else 'off'}")
    log(f"Max autonomous steps per turn: {max_agent_steps}")
    log(f"Workspace: {workspace_root_str()}")
    if history_prompt:
        log(f"History: loaded previous implementation for reference")
    log("Type /help for commands, 'quit' to exit.\n")

    pending_input = initial_message

    while True:
        if pending_input is not None:
            user_input = pending_input
            pending_input = None
            log(f"You: {user_input}")
        else:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                log("\nBye!")
                break

        if not user_input:
            continue
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
            do_compact(client, model, ctx, current_overhead_tokens=overhead_tokens)

        likely_action_intent = has_action_intent(user_input, selected_skill_names)
        autonomy_nudge = ""
        did_call_tool = False
        recovery = RecoveryState()
        suppressed_handoff_count = 0

        for step_idx in range(max_agent_steps):
            try:
                request_messages = list(ctx.messages)
                if autonomy_nudge:
                    request_messages.append({"role": "system", "content": autonomy_nudge})
                if active_skill_prompt:
                    request_messages.append(
                        {"role": "system", "content": active_skill_prompt}
                    )
                response = client.chat.completions.create(
                    model=model,
                    messages=request_messages,
                    tools=active_tool_schemas,
                )
                ctx.record_usage(response.usage, overhead_tokens=overhead_tokens)

                tool_outcome = process_tool_calls(response, ctx)
                if tool_outcome.called:
                    did_call_tool = True
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
                        (not did_call_tool)
                        or is_procedural_confirmation(content)
                        or recovery.unresolved_failure
                        or (
                            is_handoff_to_user_reply(content)
                            and suppressed_handoff_count < 2
                        )
                    )
                    and step_idx < max_agent_steps - 1
                ):
                    if recovery.unresolved_failure:
                        log("  [recovery] Suppressed unresolved-failure reply; continuing...")
                        autonomy_nudge = build_recovery_nudge(recovery)
                    elif is_handoff_to_user_reply(content):
                        suppressed_handoff_count += 1
                        log("  [autonomy] Suppressed handoff-to-user reply; continuing...")
                        autonomy_nudge = (
                            "Do not hand off to user yet. Continue autonomously and complete the task "
                            "using available tools. Only stop when done or truly blocked by missing inputs."
                        )
                    else:
                        log("  [autonomy] Suppressed non-action reply; continuing...")
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


def main():
    parser = argparse.ArgumentParser(
        description="CUDA Kernel Development Agent"
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
        "--gpu",
        type=str,
        default="auto",
        help="GPU selection: 'auto' (default) auto-detects idle GPU via rocm-smi/nvidia-smi; "
             "'none' disables GPU env injection; or an explicit index like '1'.",
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

    if not args.task:
        parser.error("--task is required (or use --list-tasks)")

    gpu_flag = (args.gpu or "").strip().lower()
    if gpu_flag == "auto":
        set_gpu_auto(True)
    elif gpu_flag and gpu_flag != "none":
        set_gpu_device(args.gpu.strip())

    # Resolve task specifier to a directory
    try:
        task_dir = resolve_task_path(args.task)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Set up run logger
    global _logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = task_dir / "logs"
    log_path = log_dir / f"{timestamp}.log"
    _logger = RunLogger(log_path)

    workdir = Path(args.workdir).resolve() if args.workdir else task_dir / "workdir"
    log(f"Setting up workspace: task={task_dir}, workdir={workdir}, gpu={args.gpu}")
    workspace = setup_workspace(task_dir, workdir)
    set_workspace_root(workspace)
    log(f"Workspace ready:\n{workspace_summary()}")
    if gpu_flag == "auto":
        log(f"GPU status: {gpu_status_summary()}")

    # Load history for this task if available
    history_prompt = load_history_prompt(task_dir)
    if history_prompt:
        log(f"Loaded history from {task_dir / 'history'}")
    log()

    initial_message = "Optimize the PyTorch model in model.py by implementing custom CUDA kernels."

    try:
        chat(
            client=client,
            model=model,
            max_tokens=args.max_tokens,
            max_agent_steps=max(1, args.max_agent_steps),
            safe_shell=args.safe_shell,
            recovery_cleanup=not args.keep_recovery_trace,
            initial_message=initial_message,
            task_dir=task_dir,
            history_prompt=history_prompt,
        )
    finally:
        log(f"\nLog saved to: {log_path}")
        _logger.close()


if __name__ == "__main__":
    main()
