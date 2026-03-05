"""
03_tool_use - Chatbot with tool/function calling and smart context compaction.

Builds on 02_context_management by adding:
- Tool/function calling: LLM can invoke registered tools
- Tool registry with decorator-based registration
- Smart context compaction: compresses old messages while preserving multi-turn structure

Usage:
    python chatbot.py                  # Start interactive chat
    python chatbot.py --list-models    # List available models
    python chatbot.py --max-tokens N   # Set context window size

Slash commands during chat:
    /tokens   - Show token usage statistics
    /history  - Show all messages with token estimates
    /debug    - Show full colored context (what the LLM sees)
    /compact  - Manually trigger smart context compaction
    /clear    - Reset conversation
    /help     - Show available commands

Environment variables (provider auto-detected, Ksyun takes priority):
    KSYUN_API_KEY    - Ksyun API key  (default model: mco-4)
    KSYUN_BASE_URL   - Ksyun base URL (default: https://kspmas.ksyun.com/v1/)
    INFINI_API_KEY   - InfiniAI API key  (default model: deepseek-v3)
    INFINI_BASE_URL  - InfiniAI base URL (default: https://cloud.infini-ai.com/maas/v1)
"""

import argparse
import sys
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm import create_client, list_models  # noqa: E402

from context import ContextManager
from tools import get_all_tool_schemas, execute_tool
from compactor import compact_messages

DEFAULT_MAX_TOKENS = 128_000

SYSTEM_PROMPT = """\
You are a helpful assistant with access to tools.
When a tool can help answer the user's question, use it.
Always prefer using tools over guessing."""

SLASH_COMMANDS_HELP = """Available commands:
  /tokens   - Show token usage statistics
  /history  - Show all messages with token estimates (compact)
  /debug    - Show full context with colored output (what the LLM sees)
  /compact  - Smart context compaction (compress old messages)
  /clear    - Reset conversation history
  /help     - Show this help message"""


def do_compact(client: OpenAI, model: str, ctx: ContextManager) -> None:
    """Run smart context compaction."""
    keep_recent = 6
    droppable = len(ctx.messages) - 1 - keep_recent
    if droppable < 4:
        print(
            f"  Not enough old messages to compact "
            f"({len(ctx.messages) - 1} total, need >{keep_recent + 3}).\n"
        )
        return

    estimated_before = ctx.get_context_tokens()
    local_before = ctx.estimate_messages_tokens()
    old_messages = ctx.messages[1:-keep_recent]

    print(f"  Compacting {len(old_messages)} old messages...")
    compacted = compact_messages(client, model, old_messages)

    if compacted:
        compacted_tokens = ctx.estimate_messages_tokens(
            [ctx.messages[0]] + compacted + ctx.messages[-keep_recent:]
        )
        if compacted_tokens < local_before:
            replaced, new_count = ctx.apply_compacted_messages(compacted, keep_recent)
            estimated_after = ctx.get_context_tokens()
            print(
                f"  Replaced {replaced} messages with {new_count} compacted messages. "
                f"Context: ~{estimated_before:,} → ~{estimated_after:,} tokens\n"
            )
        else:
            print("  Compacted version isn't smaller — skipped.\n")
    else:
        print("  Compaction failed — context unchanged.\n")


def handle_slash_command(
    command: str, client: OpenAI, model: str, ctx: ContextManager
) -> bool:
    """Handle a slash command. Returns True if handled."""
    cmd = command.strip().lower()

    if cmd == "/help":
        print(f"\n{SLASH_COMMANDS_HELP}\n")
        return True
    if cmd == "/tokens":
        print(f"\n{ctx.format_stats()}\n")
        return True
    if cmd == "/history":
        print(f"\n{ctx.format_history()}\n")
        return True
    if cmd == "/debug":
        print(f"\n{ctx.format_debug()}\n")
        return True
    if cmd == "/compact":
        do_compact(client, model, ctx)
        return True
    if cmd == "/clear":
        ctx.clear()
        print("\nConversation cleared.\n")
        return True

    return False


def format_turn_usage(usage) -> str:
    """Format the token usage from a single API response."""
    prompt = getattr(usage, "prompt_tokens", 0) or 0
    completion = getattr(usage, "completion_tokens", 0) or 0
    total = getattr(usage, "total_tokens", 0) or 0
    return f"[tokens: prompt={prompt:,}, completion={completion:,}, total={total:,}]"


def process_tool_calls(response, ctx: ContextManager) -> bool:
    """
    If the response contains tool calls, execute them and add results to context.
    Returns True if tool calls were processed (caller should continue the loop).
    """
    message = response.choices[0].message
    if not message.tool_calls:
        return False

    ctx.add_assistant_tool_calls(message)

    for tool_call in message.tool_calls:
        name = tool_call.function.name
        args = tool_call.function.arguments
        print(f"  → Calling tool: {name}({args})")

        result = execute_tool(name, args)
        print(f"  ← Result: {result}")

        ctx.add_tool_result(tool_call.id, name, result)

    return True


def chat(client: OpenAI, model: str, max_tokens: int) -> None:
    """Run an interactive multi-turn chat session with tools."""
    ctx = ContextManager(system_prompt=SYSTEM_PROMPT, max_tokens=max_tokens)
    tool_schemas = get_all_tool_schemas()

    print(f"Chatbot ready (model: {model}, context: {max_tokens:,} tokens).")
    print(f"Tools loaded: {', '.join(t['function']['name'] for t in tool_schemas)}")
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
            handle_slash_command(user_input, client, model, ctx)
            continue

        ctx.add_user_message(user_input)

        if ctx.needs_compaction():
            print("  (Context approaching limit, auto-compacting...)")
            do_compact(client, model, ctx)

        max_tool_rounds = 10
        for _ in range(max_tool_rounds):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=ctx.messages,
                    tools=tool_schemas,
                )
                ctx.record_usage(response.usage)

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
            print("  (Reached max tool call rounds)\n")


def main():
    parser = argparse.ArgumentParser(
        description="LLM Chatbot with Tool Use and Context Management"
    )
    parser.add_argument(
        "--list-models", action="store_true", help="List available models and exit"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use (default: provider-specific)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Context window size in tokens (default: {DEFAULT_MAX_TOKENS:,})",
    )
    args = parser.parse_args()

    client, provider_model = create_client()
    model = args.model or provider_model

    if args.list_models:
        list_models(client)
        return

    chat(client, model, args.max_tokens)


if __name__ == "__main__":
    main()
