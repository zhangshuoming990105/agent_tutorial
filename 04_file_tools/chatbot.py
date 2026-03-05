"""
04_file_tools - Chatbot with tool calling, smart compaction, and file tools.

Builds on 03_tool_use by adding filesystem tools:
- list_directory
- read_file
- write_file
- search_files
"""

import argparse
import json
import sys
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm import create_client, list_models  # noqa: E402

from compactor import compact_messages
from context import ContextManager
from skill_manager import build_skill_prompt, load_skills, select_skills
from tools import execute_tool, get_all_tool_schemas, get_tool_schema_map

DEFAULT_MAX_TOKENS = 128_000
DEFAULT_MAX_AGENT_STEPS = 5

SYSTEM_PROMPT = """\
You are a helpful coding assistant with tool access.

Tool usage policy:
1. Prefer tools over guessing when user asks about files or code.
2. For codebase tasks, usually do: list_directory -> search_files/read_file -> write_file.
3. Always use exact tool outputs; do not fabricate file contents.
4. Keep responses concise unless user asks for details.
"""

SLASH_COMMANDS_HELP = """Available commands:
  /tokens   - Show token usage statistics
  /history  - Show all messages with token estimates (compact)
  /debug    - Show full context with colored output (what the LLM sees)
  /compact  - Smart context compaction (compress old messages)
  /skills   - Show loaded skills and currently active skills
  /skill    - Pin/unpin a skill: /skill <name> on|off
  /verbose  - Show or set verbose token diagnostics: /verbose on|off
  /clear    - Reset conversation history
  /help     - Show this help message"""


def estimate_schema_tokens(ctx: ContextManager, tool_schemas: list[dict]) -> int:
    """Rough local estimate of tool schema tokens."""
    return ctx.estimate_tokens(json.dumps(tool_schemas, ensure_ascii=False))


def estimate_skill_tokens(ctx: ContextManager, skill_prompt: str) -> int:
    """Estimate ephemeral skill-instruction prompt token cost."""
    if not skill_prompt:
        return 0
    return ctx.estimate_tokens(skill_prompt) + 4  # one ephemeral system message


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


def process_tool_calls(response, ctx: ContextManager) -> bool:
    message = response.choices[0].message
    if not message.tool_calls:
        return False

    ctx.add_assistant_tool_calls(message)
    for tool_call in message.tool_calls:
        name = (tool_call.function.name or "").strip()
        args = tool_call.function.arguments
        if not name:
            result = "Error: empty tool name returned by model."
            print(f"  <- Result: {result}")
            ctx.add_tool_result(tool_call.id, "unknown", result)
            continue
        print(f"  -> Calling tool: {name}({args})")
        result = execute_tool(name, args)
        print(f"  <- Result: {result}")
        ctx.add_tool_result(tool_call.id, name, result)
    return True


def chat(client: OpenAI, model: str, max_tokens: int, max_agent_steps: int) -> None:
    ctx = ContextManager(system_prompt=SYSTEM_PROMPT, max_tokens=max_tokens)
    all_tool_schemas = get_all_tool_schemas()
    tool_schema_map = get_tool_schema_map()
    skills = load_skills(Path(__file__).resolve().parent / "skills")
    default_selected_skills = select_skills("", skills, pinned_on=set())
    default_tool_names = [
        tool for skill in default_selected_skills for tool in skill.tools if tool in tool_schema_map
    ]
    if not default_tool_names:
        default_tool_schemas = all_tool_schemas
        default_skill_prompt = ""
        default_skill_names = []
    else:
        # Preserve order while deduplicating
        seen = set()
        ordered = []
        for name in default_tool_names:
            if name not in seen:
                seen.add(name)
                ordered.append(name)
        default_tool_schemas = [tool_schema_map[name] for name in ordered]
        default_skill_prompt = build_skill_prompt(default_selected_skills)
        default_skill_names = [s.name for s in default_selected_skills]

    runtime_state = {
        "verbose": False,
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

        ctx.add_user_message(user_input)

        selected_skills = select_skills(
            user_input, runtime_state["skills"], runtime_state["pinned_skills"]
        )
        selected_skill_names = [s.name for s in selected_skills]
        selected_tool_names = [
            tool
            for skill in selected_skills
            for tool in skill.tools
            if tool in tool_schema_map
        ]
        if selected_tool_names:
            seen = set()
            ordered = []
            for name in selected_tool_names:
                if name not in seen:
                    seen.add(name)
                    ordered.append(name)
            active_tool_schemas = [tool_schema_map[name] for name in ordered]
        else:
            active_tool_schemas = all_tool_schemas

        active_skill_prompt = build_skill_prompt(selected_skills)
        runtime_state["active_skill_names"] = selected_skill_names
        runtime_state["active_tool_schemas"] = active_tool_schemas
        runtime_state["active_skill_prompt"] = active_skill_prompt

        if selected_skill_names:
            print(f"  [skills] active: {', '.join(selected_skill_names)}")

        overhead_tokens = estimate_schema_tokens(ctx, active_tool_schemas) + estimate_skill_tokens(
            ctx, active_skill_prompt
        )
        if ctx.needs_compaction(overhead_tokens=overhead_tokens):
            print("  (Context approaching limit, auto-compacting...)")
            do_compact(client, model, ctx, current_overhead_tokens=overhead_tokens)

        for _ in range(max_agent_steps):
            try:
                request_messages = list(ctx.messages)
                if active_skill_prompt:
                    request_messages.append({"role": "system", "content": active_skill_prompt})
                response = client.chat.completions.create(
                    model=model,
                    messages=request_messages,
                    tools=active_tool_schemas,
                )
                ctx.record_usage(response.usage, overhead_tokens=overhead_tokens)

                if process_tool_calls(response, ctx):
                    continue

                content = response.choices[0].message.content or ""
                ctx.add_assistant_message(content)
                print(f"\nAssistant: {content}")
                print(f"  {format_turn_usage(response.usage)}\n")
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
            ctx.add_assistant_message(help_msg)
            print(f"\nAssistant: {help_msg}\n")


def main():
    parser = argparse.ArgumentParser(
        description="LLM Chatbot with Tool Use, File Tools, and Context Management"
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
    args = parser.parse_args()

    client, provider_model = create_client()
    model = args.model or provider_model
    if args.list_models:
        list_models(client)
        return
    chat(client, model, args.max_tokens, max(1, args.max_agent_steps))


if __name__ == "__main__":
    main()
