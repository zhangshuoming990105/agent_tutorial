"""
01_basic_chatbot - A simple multi-turn chatbot using OpenAI-compatible API.

This is the first step in building a general-purpose LLM agent.
We use the OpenAI SDK to talk to InfiniAI's API (or any OpenAI-compatible provider).

Usage:
    python chatbot.py                  # Start interactive chat
    python chatbot.py --list-models    # List available models

Environment variables:
    INFINI_API_KEY   - Your InfiniAI API key
    INFINI_BASE_URL  - API base URL (default: https://cloud.infini-ai.com/maas/v1)
"""

import argparse
import os
import sys

from openai import OpenAI

DEFAULT_BASE_URL = "https://cloud.infini-ai.com/maas/v1"
DEFAULT_MODEL = "deepseek-v3"

SYSTEM_PROMPT = "You are a helpful assistant."


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
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"Model to use (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    client = create_client()

    if args.list_models:
        list_models(client)
        return

    chat(client, args.model)


if __name__ == "__main__":
    main()
