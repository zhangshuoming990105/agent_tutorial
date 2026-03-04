# Step 02: Context Management

Building on the basic chatbot, we add **token counting** and **context inspection** — essential infrastructure for any LLM application.

## What's New

- **Token counting**: Every turn displays API-reported token usage (prompt, completion, total)
- **Local estimation**: `tiktoken` estimates token counts before sending
- **Context window tracking**: See how much of the context window is in use
- **Colored debug view**: `/debug` shows the full context exactly as the LLM sees it, color-coded by role
- **Slash commands**: `/tokens`, `/history`, `/debug`, `/clear` for inspection and control

## Setup

```bash
pip install -r requirements.txt
export INFINI_API_KEY="your-api-key-here"
```

## Usage

```bash
python chatbot.py
python chatbot.py --model qwen3-32b
python chatbot.py --max-tokens 32000
```

### Slash Commands

| Command    | Description |
|------------|-------------|
| `/tokens`  | Show cumulative token usage and context utilization |
| `/history` | Show all messages with per-message token estimates (compact) |
| `/debug`   | Show full colored context — what the LLM actually sees |
| `/clear`   | Reset the entire conversation |
| `/help`    | List available commands |

## Key Concepts

### Two Kinds of Token Counting

1. **API-reported** (`response.usage`): Exact count from the server after each request. Ground truth, but only available *after* the call.

2. **Local estimation** (`tiktoken`): Approximate count computed locally *before* sending. We use `cl100k_base` encoding as a cross-model approximation — not exact for DeepSeek or Qwen tokenizers, but close enough for budgeting.

### Why This Matters for Agents

In later steps, agents will make many LLM calls per user request (tool calls, reasoning loops). Understanding token consumption is critical:

- Tool call results can be large (file contents, command output)
- Multi-step reasoning accumulates many messages quickly
- Without tracking, you can't tell when you're about to hit the context limit

Context *compaction* (intelligently summarizing old messages) will be added in Step 03 alongside tool use, where it becomes truly necessary.

## Files

- `chatbot.py` — Main entry point with REPL and slash commands
- `context.py` — `ContextManager` class: token counting, stats, colored debug output

## What's Next

In [Step 03](../03_tool_use/), we'll add tool/function calling and smart context compaction — letting the LLM invoke external functions and intelligently compress conversation history.
