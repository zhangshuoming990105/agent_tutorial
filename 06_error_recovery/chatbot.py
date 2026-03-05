"""
06_error_recovery - Adds task-level error recovery on top of shell/file tools.

New in this step:
- task-level recovery state for autonomous multi-step execution
- structured retries guided by latest tool failures
- loop guard against premature plan/confirm/handoff replies
- optional cleanup of failed intermediate traces after task success
- run_shell cwd parameter (no more manual cd)
- system prompt with workspace root for path awareness
- workspace hint extraction from user input
"""

import argparse
from dataclasses import dataclass, field
import json
import re
import sys
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm import create_client, list_models  # noqa: E402

from compactor import compact_messages
from context import ContextManager
from skill_manager import build_skill_prompt, load_skills, select_skills
from tools import (
    execute_tool,
    get_all_tool_schemas,
    get_shell_policy_snapshot,
    get_tool_schema_map,
    set_shell_safety,
    workspace_root_str,
)

DEFAULT_MAX_TOKENS = 128_000
DEFAULT_MAX_AGENT_STEPS = 5

_SYSTEM_PROMPT_TEMPLATE = """\
You are a helpful coding assistant with tool access.

Workspace root: {workspace_root}
All file-tool paths are relative to this root. run_shell accepts a `cwd` parameter \
(relative to workspace root) so you never need to `cd` manually.

Tool usage policy:
1. Prefer tools over guessing for file/code/shell tasks.
2. For codebase tasks, usually do: list_directory -> grep_text/search_files/read_file -> write_file.
3. For shell tasks, use run_shell with an appropriate cwd; do not fabricate shell outputs.
4. Keep responses concise unless user asks for details.
5. When compiling/running code, ensure output formats match exactly before declaring success.
6. Fix all compiler warnings before declaring a translation/port is complete.
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
  /workspace    - Show current per-turn workspace hint
  /clear        - Reset conversation history
  /help         - Show this help message"""

CONTINUATION_REPLIES = {
    "y",
    "yes",
    "ok",
    "okay",
    "sure",
    "continue",
    "go",
    "proceed",
    "继续",
    "好的",
    "好",
}
ACTION_INTENT_RE = re.compile(
    r"\b(run|execute|check|list|read|write|search|find|show|cat|grep|ls|pwd)\b"
)
AUTONOMY_NUDGE_TEXT = (
    "Do not provide a plan or ask the user for procedural confirmation. "
    "For this turn, immediately call the required tools and only then provide a final summary."
)
EXIT_CODE_RE = re.compile(r"^exit_code=(-?\d+)$", re.MULTILINE)
WORKSPACE_HINT_RE = re.compile(r"workspace\s+is\s+([^\s,;.]+(?:/[^\s,;.]+)*)", re.IGNORECASE)
HANDOFF_PHRASES = (
    "let me know",
    "if you'd like",
    "would you like",
    "please provide",
    "to proceed",
    "share the file",
    "share the code",
)


@dataclass
class ToolExecutionOutcome:
    called: bool = False
    failures: list[str] = field(default_factory=list)


@dataclass
class RecoveryState:
    had_failure: bool = False
    unresolved_failure: bool = False
    repeated_failure_count: int = 0
    last_failure_signature: str = ""
    failures: list[str] = field(default_factory=list)

    def record_failures(self, failures: list[str]) -> None:
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
    workspace_prompt: str = "",
    verbose: bool = False,
) -> str:
    schema_tokens = estimate_schema_tokens(ctx, tool_schemas)
    skill_tokens = estimate_skill_tokens(ctx, skill_prompt) + estimate_skill_tokens(
        ctx, workspace_prompt
    )
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
                runtime_state.get("active_workspace_prompt", ""),
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
        ) + estimate_skill_tokens(
            ctx, runtime_state["active_skill_prompt"]
        ) + estimate_skill_tokens(ctx, runtime_state.get("active_workspace_prompt", ""))
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
        hint = runtime_state.get("workspace_hint", "") or "(none)"
        print(f"\nWorkspace hint: {hint}\n")
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
        "proceed?",
        "reply `yes` or `no`",
        "reply yes or no",
        "wait for your go-ahead",
        "action: proceed",
        "approve",
        "approval",
        "go-ahead",
        "allow once",
        "always allow",
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
    return bool(ACTION_INTENT_RE.search(user_input.lower())) or ("shell" in selected_skill_names)


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
    latest = recovery.failures[-1] if recovery.failures else "unknown failure"
    repeat_hint = ""
    if recovery.repeated_failure_count >= 2:
        repeat_hint = (
            "You repeated a similar failing attempt. Change strategy and do not retry the same "
            "command/arguments unchanged."
        )
    return (
        "Recovery mode: latest tool execution failed.\n"
        f"- Latest failure: {latest}\n"
        "- Diagnose root cause from tool output and perform a concrete fix in the next tool call.\n"
        "- If needed, inspect files or environment before retrying.\n"
        "- Do not ask user for procedural confirmation; continue autonomously within this turn.\n"
        f"{repeat_hint}"
    ).strip()


def extract_workspace_hint(user_input: str) -> str | None:
    m = WORKSPACE_HINT_RE.search(user_input)
    if not m:
        return None
    hint = m.group(1).strip().strip("`'\"")
    return hint or None


def build_workspace_prompt(workspace_hint: str | None) -> str:
    if not workspace_hint:
        return ""
    return (
        "Task workspace hint:\n"
        f"- Use `{workspace_hint}` as the primary working directory for file paths in this turn.\n"
        "- Prefer absolute paths under that workspace when calling tools."
    )


def is_handoff_to_user_reply(text: str) -> bool:
    low = (text or "").lower()
    if any(p in low for p in HANDOFF_PHRASES):
        return True
    # Questions that sound like procedural handoff usually end a turn prematurely.
    return low.strip().endswith("?") and ("you" in low or "your" in low)


def process_tool_calls(response, ctx: ContextManager) -> ToolExecutionOutcome:
    message = response.choices[0].message
    if not message.tool_calls:
        return ToolExecutionOutcome(called=False, failures=[])

    ctx.add_assistant_tool_calls(message)
    failures: list[str] = []
    for tool_call in message.tool_calls:
        name = (tool_call.function.name or "").strip()
        args = tool_call.function.arguments
        if not name:
            result = "Error: empty tool name returned by model."
            print(f"  <- Result: {result}")
            ctx.add_tool_result(tool_call.id, "unknown", result)
            failures.append("unknown: empty tool name")
            continue
        print(f"  -> Calling tool: {name}({args})")
        result = execute_tool(name, args)
        print(f"  <- Result: {result}")
        ctx.add_tool_result(tool_call.id, name, result)
        if is_tool_failure(name, result):
            failures.append(summarize_failure(name, result))
    return ToolExecutionOutcome(called=True, failures=failures)


def chat(
    client: OpenAI,
    model: str,
    max_tokens: int,
    max_agent_steps: int,
    safe_shell: bool,
    recovery_cleanup: bool,
) -> None:
    set_shell_safety(safe_shell)

    system_prompt = get_system_prompt()
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
        "workspace_hint": "",
        "active_workspace_prompt": "",
        "skills": skills,
        "pinned_skills": set(),
        "active_skill_names": default_skill_names,
        "active_tool_schemas": default_tool_schemas,
        "active_skill_prompt": default_skill_prompt,
    }

    print(f"Chatbot ready (model: {model}, context: {max_tokens:,} tokens).")
    print(f"Tools loaded: {', '.join(t['function']['name'] for t in all_tool_schemas)}")
    if skills:
        print(f"Skills loaded: {', '.join(sorted(skills.keys()))}")
    print(f"Safe shell mode: {'on' if safe_shell else 'off'}")
    print(f"Recovery cleanup: {'on' if recovery_cleanup else 'off'}")
    print(f"Max autonomous steps per turn: {max_agent_steps}")
    print("Type /help for commands, 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        if user_input.startswith("/"):
            handle_slash_command(user_input, client, model, ctx, runtime_state)
            continue

        turn_start_index = len(ctx.messages)
        ctx.add_user_message(user_input)
        workspace_hint = extract_workspace_hint(user_input)
        if workspace_hint:
            runtime_state["workspace_hint"] = workspace_hint

        selected_skills = select_skills(
            user_input, runtime_state["skills"], runtime_state["pinned_skills"]
        )
        # For short continuation replies, preserve prior active skills so loops can continue.
        maybe_extend_skills_for_continuation(user_input, selected_skills, runtime_state)
        selected_skill_names = [s.name for s in selected_skills]
        active_tool_schemas = resolve_active_tool_schemas(
            selected_skills, tool_schema_map, all_tool_schemas
        )

        active_skill_prompt = build_skill_prompt(selected_skills)
        active_workspace_prompt = build_workspace_prompt(runtime_state["workspace_hint"])
        runtime_state["active_skill_names"] = selected_skill_names
        runtime_state["active_tool_schemas"] = active_tool_schemas
        runtime_state["active_skill_prompt"] = active_skill_prompt
        runtime_state["active_workspace_prompt"] = active_workspace_prompt

        if selected_skill_names:
            print(f"  [skills] active: {', '.join(selected_skill_names)}")

        overhead_tokens = estimate_schema_tokens(
            ctx, active_tool_schemas
        ) + estimate_skill_tokens(ctx, active_skill_prompt) + estimate_skill_tokens(
            ctx, active_workspace_prompt
        )
        if ctx.needs_compaction(overhead_tokens=overhead_tokens):
            print("  (Context approaching limit, auto-compacting...)")
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
                if active_workspace_prompt:
                    request_messages.append(
                        {"role": "system", "content": active_workspace_prompt}
                    )
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
                        recovery.record_failures(tool_outcome.failures)
                        print(f"  [recovery] Detected failure: {tool_outcome.failures[-1]}")
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
                        print("  [recovery] Suppressed unresolved-failure reply; continuing...")
                        autonomy_nudge = build_recovery_nudge(recovery)
                    elif is_handoff_to_user_reply(content):
                        suppressed_handoff_count += 1
                        print("  [autonomy] Suppressed handoff-to-user reply; continuing...")
                        autonomy_nudge = (
                            "Do not hand off to user yet. Continue autonomously and complete the task "
                            "using available tools. Only stop when done or truly blocked by missing inputs."
                        )
                    else:
                        print("  [autonomy] Suppressed non-action reply; continuing...")
                        autonomy_nudge = AUTONOMY_NUDGE_TEXT
                    continue

                ctx.add_assistant_message(content)
                print(f"\nAssistant: {content}")
                print(f"  {format_turn_usage(response.usage)}\n")
                if runtime_state.get("recovery_cleanup", True) and recovery.had_failure:
                    removed = ctx.drop_failed_tool_messages(start_index=turn_start_index + 1)
                    if removed > 0:
                        print(f"  [recovery] Cleaned {removed} failed intermediate messages from context.\n")
                break
            except Exception as e:
                print(f"\nError: {e}\n")
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
            print(f"\nAssistant: {help_msg}\n")


def main():
    parser = argparse.ArgumentParser(
        description="LLM Chatbot with task-level error recovery"
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
        "--unsafe-shell",
        action="store_true",
        help="Disable safe shell mode (NOT recommended).",
    )
    parser.add_argument(
        "--keep-recovery-trace",
        action="store_true",
        help="Keep failed intermediate traces in context after successful task completion.",
    )
    args = parser.parse_args()

    client, provider_model = create_client()
    model = args.model or provider_model
    if args.list_models:
        list_models(client)
        return

    chat(
        client=client,
        model=model,
        max_tokens=args.max_tokens,
        max_agent_steps=max(1, args.max_agent_steps),
        safe_shell=not args.unsafe_shell,
        recovery_cleanup=not args.keep_recovery_trace,
    )


if __name__ == "__main__":
    main()
