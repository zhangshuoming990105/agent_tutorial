# Step 03: Tool Use

The LLM can now **call external functions** — the foundation of agentic behavior. We also add **smart context compaction** and **calibrated token tracking** based on API-reported usage.

## What's New

- **Tool registry**: Decorator-based system to register Python functions as LLM-callable tools
- **Tool execution loop**: LLM returns tool calls → parse → execute → feed results back → LLM continues (this is already an agent loop)
- **Demo tools**: `calculator` (math expressions) and `get_current_time`
- **Smart compaction**: `/compact` uses the LLM to compress old messages into fewer, denser ones while keeping the user/assistant alternating format
- **Calibrated token tracking**: Uses API-reported `prompt_tokens` as baseline + tiktoken delta for new messages, instead of pure local estimation (which misses tool schema overhead and tokenizer differences)

## Architecture

```
User ──> LLM API (with tool schemas)
             │
             ├── text response ──> display to user
             │
             └── tool_calls ──> execute tools ──> tool results
                                                       │
                                                       └──> back to LLM (loop)
```

The tool call loop is already an autonomous agent loop — the LLM can chain multiple tool calls and reasoning steps per user message, up to `max_tool_rounds`.

## Setup

```bash
pip install -r requirements.txt
export INFINI_API_KEY="your-api-key-here"
python chatbot.py
```

## Key Concepts

### How the LLM Knows Which Tools to Call

Tool schemas (name, description, parameters) are passed to the API via the `tools=` parameter on every call. The LLM sees these schemas as part of its prompt and decides whether to invoke them based on the user's request. This is why tool schemas cost tokens — the API serializes them into the prompt in an expanded format that's ~3-4x larger than the raw JSON.

### Tool Registration

Tools are registered via a decorator that pairs a Python function with its JSON schema:

```python
@tool(
    name="calculator",
    description="Evaluate a mathematical expression.",
    parameters={
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "e.g. '2 ** 10'"}
        },
        "required": ["expression"],
    },
)
def calculator(expression: str) -> str:
    ...
```

The registry automatically collects all schemas to pass to the API.

### Tool Call Flow

1. Send messages + tool schemas to LLM
2. If response contains `tool_calls`, execute each tool
3. Add tool results as `role: "tool"` messages
4. Call LLM again (it sees the results and can call more tools or respond)
5. Repeat until LLM responds with text (no more tool calls)

### Calibrated Token Tracking

Pure tiktoken estimation can be off by 90%+ because:
- Tool schemas are serialized into the prompt in an expanded format (~3-4x the raw JSON)
- Different models use different tokenizers (`cl100k_base` is for GPT-4, not DeepSeek/Qwen)
- APIs may inject hidden system instructions

Our solution: use the last API-reported `prompt_tokens` as a calibrated baseline, then add tiktoken deltas for any new messages. This brings error down to ~3%.

### Smart Compaction

Unlike simple truncation or paragraph summarization, our compactor:

- Maintains the multi-turn `user → assistant` structure
- Keeps short user messages verbatim
- Condenses assistant responses to key deliverables (code, data, results)
- Collapses tool call sequences into brief notes (e.g. "Used calculator: 2^10 = 1024")
- Drops reasoning chains and fluff
- Only replaces old messages if the result is actually smaller

## Files

- `chatbot.py` — Main entry point with tool calling loop
- `context.py` — ContextManager with calibrated token tracking, tool message support, and compaction
- `tools.py` — Tool registry and demo tools (calculator, get_current_time)
- `compactor.py` — Smart LLM-powered context compaction

## What's Next

In [Step 04](../04_file_tools/), we'll add file system tools (read, write, list, search) so the agent can interact with codebases — the first step toward a real coding agent.
