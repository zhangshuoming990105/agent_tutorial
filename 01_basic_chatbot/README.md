# Step 01: Basic Chatbot

The simplest possible starting point — call an LLM and have a multi-turn conversation.

## What you'll learn

- How to use the OpenAI SDK with an OpenAI-compatible provider (InfiniAI)
- How to maintain multi-turn conversation history via a `messages` list
- How to configure system prompts

## Architecture

```
User Input ──> messages list ──> LLM API ──> Assistant Response
                   ▲                              │
                   └──────────────────────────────┘
                        (append to history)
```

The key data structure is the `messages` list — an ordered sequence of role-tagged messages (`system`, `user`, `assistant`) that represents the full conversation context sent to the LLM on every call.

## Setup

```bash
pip install -r requirements.txt
export INFINI_API_KEY="your-api-key-here"
```

The default base URL points to InfiniAI (`https://cloud.infini-ai.com/maas/v1`). Override with:

```bash
export INFINI_BASE_URL="https://your-provider.com/v1"
```

## Usage

**List available models:**

```bash
python chatbot.py --list-models
```

**Start chatting (default model: deepseek-v3):**

```bash
python chatbot.py
```

**Use a specific model:**

```bash
python chatbot.py --model qwen3-32b
```

## Key Concepts

### The Messages List

Every LLM API call is **stateless** — the model has no memory. Multi-turn conversation works by sending the entire conversation history each time:

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a programming language..."},
    {"role": "user", "content": "How do I install it?"},  # new turn
]
```

This is the fundamental pattern that all LLM applications build upon. In later steps, we'll need to manage this list carefully as conversations grow long.

### OpenAI-Compatible API

The OpenAI SDK can talk to any provider that implements the same API format. We just change `base_url`:

```python
client = OpenAI(api_key="...", base_url="https://cloud.infini-ai.com/maas/v1")
```

This means our code works with OpenAI, InfiniAI, Ollama, vLLM, and many others.

## What's Next

In [Step 02](../02_context_management/), we'll add token counting and context window management — essential infrastructure before we start building agent capabilities.
