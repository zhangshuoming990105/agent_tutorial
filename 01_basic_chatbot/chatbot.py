"""
01_basic_chatbot - A simple multi-turn chatbot using OpenAI-compatible API.

This is the first step in building a general-purpose LLM agent.
We use the OpenAI SDK to talk to any OpenAI-compatible provider.

Usage:
    python chatbot.py                  # Start interactive chat
    python chatbot.py --list-models    # List available models

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

SYSTEM_PROMPT = "You are a helpful assistant."


def chat(client: OpenAI, model: str) -> None:
    """Run an interactive multi-turn chat session."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print(f"Chatbot ready (model: {model}). Type 'quit' to exit.\n")

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

        messages.append({"role": "user", "content": user_input})

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
            )
            assistant_message = response.choices[0].message
            content = assistant_message.content or ""
            messages.append({"role": "assistant", "content": content})

            print(f"\nAssistant: {content}\n")

        except Exception as e:
            print(f"\nError: {e}\n")
            messages.pop()


def main():
    parser = argparse.ArgumentParser(description="Simple LLM Chatbot")
    parser.add_argument("--list-models", action="store_true", help="List available models and exit")
    parser.add_argument("--model", type=str, default=None, help="Model to use (default: provider-specific)")
    args = parser.parse_args()

    client, provider_model = create_client()
    model = args.model or provider_model

    if args.list_models:
        list_models(client)
        return

    chat(client, model)


if __name__ == "__main__":
    main()
