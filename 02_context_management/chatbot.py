"""
02_context_management - Chatbot with token counting and context inspection.

Builds on 01_basic_chatbot by adding:
- Token counting (API-reported + local estimation via tiktoken)
- Per-turn usage display
- Slash commands for inspecting context state
- Colored debug view of the full LLM context

Usage:
    python chatbot.py                  # Start interactive chat
    python chatbot.py --list-models    # List available models
    python chatbot.py --max-tokens N   # Set context window size

Slash commands during chat:
    /tokens   - Show token usage statistics
    /history  - Show all messages with token estimates
    /debug    - Show full colored context (what the LLM sees)
    /clear    - Reset conversation
    /help     - Show available commands

Environment variables:
    INFINI_API_KEY   - Your InfiniAI API key
    INFINI_BASE_URL  - API base URL (default: https://cloud.infini-ai.com/maas/v1)
"""

import argparse
import os
import sys

from openai import OpenAI

from context import ContextManager

DEFAULT_BASE_URL = "https://cloud.infini-ai.com/maas/v1"
DEFAULT_MODEL = "deepseek-v3"
DEFAULT_MAX_TOKENS = 128_000

SYSTEM_PROMPT = "You are a helpful assistant."

SLASH_COMMANDS_HELP = """Available commands:
  /tokens   - Show token usage statistics
  /history  - Show all messages with token estimates (compact)
  /debug    - Show full context with colored output (what the LLM sees)
  /clear    - Reset conversation history
  /help     - Show this help message"""


def create_client() -> OpenAI:
    api_key = os.getenv("INFINI_API_KEY")
    base_url = os.getenv("INFINI_BASE_URL", DEFAULT_BASE_URL)

    if not api_key:
        print("Error: INFINI_API_KEY environment variable is not set.")
        print("Get your API key from https://cloud.infini-ai.com")
        sys.exit(1)

    return OpenAI(api_key=api_key, base_url=base_url)


def list_models(client: OpenAI) -> None:
    """Fetch and display available models from the API."""
    print("Fetching available models...\n")
    try:
        models = client.models.list()
        model_list = sorted(models.data, key=lambda m: m.id)
        print(f"Found {len(model_list)} models:\n")
        for m in model_list:
            print(f"  - {m.id}")
        print()
    except Exception as e:
        print(f"Failed to list models: {e}")
        sys.exit(1)


def handle_slash_command(command: str, ctx: ContextManager) -> bool:
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


def chat(client: OpenAI, model: str, max_tokens: int) -> None:
    """Run an interactive multi-turn chat session with context tracking."""
    ctx = ContextManager(system_prompt=SYSTEM_PROMPT, max_tokens=max_tokens)

    print(f"Chatbot ready (model: {model}, context: {max_tokens:,} tokens).")
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
            handle_slash_command(user_input, ctx)
            continue

        ctx.add_user_message(user_input)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=ctx.messages,
            )
            assistant_message = response.choices[0].message
            content = assistant_message.content or ""
            ctx.add_assistant_message(content)
            ctx.record_usage(response.usage)

            print(f"\nAssistant: {content}")
            print(f"  {format_turn_usage(response.usage)}\n")

        except Exception as e:
            print(f"\nError: {e}\n")
            ctx.pop_last_message()


def main():
    parser = argparse.ArgumentParser(description="LLM Chatbot with Context Management")
    parser.add_argument(
        "--list-models", action="store_true", help="List available models and exit"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Context window size in tokens (default: {DEFAULT_MAX_TOKENS:,})",
    )
    args = parser.parse_args()

    client = create_client()

    if args.list_models:
        list_models(client)
        return

    chat(client, args.model, args.max_tokens)


if __name__ == "__main__":
    main()
