"""
Non-interactive test harness: feeds a single prompt into the agent loop directly,
bypassing input() entirely.  All intermediate output (tool calls, results, LLM
replies, suppression events) is printed with flush=True for real-time visibility.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Force line-buffered stdout so every print() is visible immediately in terminal.
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(__file__))

from context import ContextManager
from skill_manager import build_skill_prompt, load_skills, select_skills
from tools import (
    execute_tool,
    get_all_tool_schemas,
    get_tool_schema_map,
    set_shell_safety,
)
from chatbot import (
    AUTONOMY_NUDGE_TEXT,
    get_system_prompt,
    RecoveryState,
    ToolExecutionOutcome,
    build_recovery_nudge,
    build_workspace_prompt,
    create_client,
    estimate_schema_tokens,
    estimate_skill_tokens,
    extract_workspace_hint,
    format_turn_usage,
    has_action_intent,
    is_handoff_to_user_reply,
    is_procedural_confirmation,
    is_tool_failure,
    resolve_active_tool_schemas,
    summarize_failure,
)

C = {
    "R": "\033[0m",
    "B": "\033[1m",
    "D": "\033[2m",
    "RED": "\033[31m",
    "GRN": "\033[32m",
    "YEL": "\033[33m",
    "BLU": "\033[34m",
    "MAG": "\033[35m",
    "CYN": "\033[36m",
}


def p(msg: str = "", **kwargs) -> None:
    print(msg, flush=True, **kwargs)


def p_header(label: str, value: str) -> None:
    p(f"  {C['D']}{label:18s}{C['R']} {value}")


def p_sep() -> None:
    p(f"{C['D']}{'─' * 72}{C['R']}")


def process_tool_calls_verbose(response, ctx: ContextManager) -> ToolExecutionOutcome:
    message = response.choices[0].message
    if not message.tool_calls:
        return ToolExecutionOutcome(called=False, failures=[])

    ctx.add_assistant_tool_calls(message)
    failures: list[str] = []

    for tc in message.tool_calls:
        name = (tc.function.name or "").strip()
        raw_args = tc.function.arguments or ""
        if not name:
            result = "Error: empty tool name returned by model."
            p(f"  {C['RED']}✗ (empty tool name){C['R']}  {result}")
            ctx.add_tool_result(tc.id, "unknown", result)
            failures.append("unknown: empty tool name")
            continue

        try:
            pretty_args = json.dumps(json.loads(raw_args), ensure_ascii=False)
        except Exception:
            pretty_args = raw_args

        p(f"  {C['CYN']}→ {name}{C['R']}({pretty_args})")

        t0 = time.time()
        result = execute_tool(name, raw_args)
        elapsed = time.time() - t0

        lines = result.strip().splitlines()
        preview_lines = 12
        is_failure = is_tool_failure(name, result)
        color = C["RED"] if is_failure else C["GRN"]
        status = "✗ FAIL" if is_failure else "✓ OK"

        p(f"  {color}← {status}{C['R']}  ({elapsed:.1f}s, {len(lines)} lines)")
        for line in lines[:preview_lines]:
            p(f"    {C['D']}{line}{C['R']}")
        if len(lines) > preview_lines:
            p(f"    {C['D']}... ({len(lines) - preview_lines} more lines){C['R']}")

        ctx.add_tool_result(tc.id, name, result)
        if is_failure:
            failures.append(summarize_failure(name, result))

    return ToolExecutionOutcome(called=True, failures=failures)


def run_single_turn(user_input: str, max_steps: int = 25):
    client = create_client()
    model = os.getenv("TEST_MODEL", "deepseek-v3")
    max_tokens = 128_000

    set_shell_safety(False)

    system_prompt = get_system_prompt()
    ctx = ContextManager(system_prompt=system_prompt, max_tokens=max_tokens)
    all_tool_schemas = get_all_tool_schemas()
    tool_schema_map = get_tool_schema_map()
    skills = load_skills(Path(__file__).resolve().parent / "skills")

    ctx.add_user_message(user_input)

    selected_skills = select_skills(user_input, skills, pinned_on=set())
    selected_skill_names = [s.name for s in selected_skills]
    active_tool_schemas = resolve_active_tool_schemas(
        selected_skills, tool_schema_map, all_tool_schemas
    )
    active_skill_prompt = build_skill_prompt(selected_skills)

    workspace_hint = extract_workspace_hint(user_input)
    workspace_prompt = build_workspace_prompt(workspace_hint)

    overhead_tokens = (
        estimate_schema_tokens(ctx, active_tool_schemas)
        + estimate_skill_tokens(ctx, active_skill_prompt)
        + estimate_skill_tokens(ctx, workspace_prompt)
    )

    p_sep()
    p(f"{C['B']}Agent Test Run{C['R']}")
    p_sep()
    p_header("Model:", model)
    p_header("Skills:", ", ".join(selected_skill_names))
    p_header("Workspace hint:", str(workspace_hint))
    p_header("Max steps:", str(max_steps))
    p_header("Overhead:", f"~{overhead_tokens} tokens")
    p_sep()
    p(f"\n{C['GRN']}User:{C['R']} {user_input}\n")

    likely_action_intent = has_action_intent(user_input, selected_skill_names)
    autonomy_nudge = ""
    did_call_tool = False
    recovery = RecoveryState()
    suppressed_handoff_count = 0
    t_start = time.time()

    for step_idx in range(max_steps):
        step_label = f"Step {step_idx + 1}/{max_steps}"
        p(f"\n{C['B']}━━━ {step_label} ━━━{C['R']}")

        try:
            request_messages = list(ctx.messages)
            if autonomy_nudge:
                request_messages.append({"role": "system", "content": autonomy_nudge})
            if workspace_prompt:
                request_messages.append({"role": "system", "content": workspace_prompt})
            if active_skill_prompt:
                request_messages.append({"role": "system", "content": active_skill_prompt})

            p(f"  {C['D']}Calling LLM ({len(request_messages)} messages)...{C['R']}")
            t0 = time.time()
            response = client.chat.completions.create(
                model=model,
                messages=request_messages,
                tools=active_tool_schemas,
            )
            llm_elapsed = time.time() - t0
            usage = response.usage
            p(
                f"  {C['D']}LLM responded in {llm_elapsed:.1f}s "
                f"(prompt={getattr(usage, 'prompt_tokens', 0)}, "
                f"completion={getattr(usage, 'completion_tokens', 0)}){C['R']}"
            )
            ctx.record_usage(usage, overhead_tokens=overhead_tokens)

            tool_outcome = process_tool_calls_verbose(response, ctx)
            if tool_outcome.called:
                did_call_tool = True
                if tool_outcome.failures:
                    recovery.record_failures(tool_outcome.failures)
                    p(
                        f"  {C['RED']}[recovery] "
                        f"Failure detected: {tool_outcome.failures[-1]}{C['R']}"
                    )
                    autonomy_nudge = build_recovery_nudge(recovery)
                else:
                    recovery.unresolved_failure = False
                    autonomy_nudge = ""
                continue

            content = response.choices[0].message.content or ""
            should_suppress = False
            reason = ""

            if likely_action_intent and step_idx < max_steps - 1:
                if not did_call_tool:
                    should_suppress = True
                    reason = "no tool called yet"
                    autonomy_nudge = AUTONOMY_NUDGE_TEXT
                elif is_procedural_confirmation(content):
                    should_suppress = True
                    reason = "procedural confirmation"
                    autonomy_nudge = AUTONOMY_NUDGE_TEXT
                elif recovery.unresolved_failure:
                    should_suppress = True
                    reason = "unresolved failure"
                    autonomy_nudge = build_recovery_nudge(recovery)
                elif (
                    is_handoff_to_user_reply(content)
                    and suppressed_handoff_count < 2
                ):
                    should_suppress = True
                    suppressed_handoff_count += 1
                    reason = f"handoff to user (#{suppressed_handoff_count})"
                    autonomy_nudge = (
                        "Do not hand off to user yet. Continue autonomously and "
                        "complete the task using available tools. Only stop when "
                        "done or truly blocked by missing inputs."
                    )

            if should_suppress:
                preview = content[:300].replace("\n", "\\n")
                p(f"  {C['YEL']}[suppressed: {reason}]{C['R']}")
                p(f"  {C['D']}Preview: {preview}{C['R']}")
                continue

            ctx.add_assistant_message(content)
            p_sep()
            p(f"\n{C['BLU']}{C['B']}Final Response:{C['R']}\n")
            p(content)
            p(f"\n  {format_turn_usage(usage)}")
            break
        except Exception as e:
            p(f"\n  {C['RED']}ERROR: {e}{C['R']}")
            ctx.pop_last_message()
            break
    else:
        p(f"\n{C['RED']}--- MAX STEPS EXHAUSTED ({max_steps}) ---{C['R']}")

    elapsed_total = time.time() - t_start
    p_sep()
    p(f"{C['B']}Summary{C['R']}")
    p_header("Messages:", str(len(ctx.messages)))
    p_header("API requests:", str(ctx.stats.total_requests))
    p_header("Total tokens:", f"{ctx.stats.total_tokens:,}")
    p_header("Wall time:", f"{elapsed_total:.1f}s")
    p_sep()


if __name__ == "__main__":
    prompt = (
        "workspace is 06_error_recovery/task/code_translation. "
        "Read example.c, translate it to Rust as main.rs in that same directory. "
        "Compile and run both the C and Rust versions. "
        "Compare outputs line by line. If mismatch, fix the Rust code and retry. "
        "Finally explain whether the calculation steps are reasonable."
    )
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
    run_single_turn(prompt, max_steps=25)
