"""
Smart context compaction.

Compresses conversation history while preserving multi-turn structure.
Instead of collapsing into a single summary paragraph, produces a compressed
sequence of user/assistant messages where:
- Short user messages are kept verbatim
- Long user messages are condensed to core intent
- Assistant responses keep only key content (code, data, conclusions)
- Reasoning chains and fluff are stripped
- Failed attempts are dropped entirely
- Tool calls are condensed to "called X → result summary"
"""

from __future__ import annotations

import json

from openai import OpenAI

COMPACTOR_SYSTEM_PROMPT = """\
You are a conversation compressor. Given a conversation history, produce a \
shorter version that preserves the essential information.

Rules:
1. Output MUST be a valid JSON array of message objects: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
2. Maintain the multi-turn structure (alternating user/assistant messages).
3. For user messages: keep short ones (<80 chars) verbatim. For longer ones, condense to the core request/question.
4. For assistant messages: keep ONLY the key deliverables — code blocks, data, direct answers, conclusions. Remove greetings, encouragement, lengthy explanations, and thinking/reasoning chains.
5. For tool call sequences: condense to a brief note like "Used calculator: 2^10 = 1024" inside the assistant message.
6. Drop failed attempts or error-recovery loops entirely — only keep the final successful result.
7. Merge adjacent messages of the same role if it makes the conversation cleaner.
8. The compressed version should be roughly 30-50% of the original token count.
9. Write in the same language as the original conversation.
10. Output ONLY the JSON array, no markdown fences, no explanation."""


def compact_messages(
    client: OpenAI,
    model: str,
    messages_to_compact: list[dict],
    max_output_tokens: int = 1024,
) -> list[dict] | None:
    """
    Use the LLM to compress a list of messages into fewer, denser messages.

    Returns a list of compacted message dicts, or None on failure.
    """
    if len(messages_to_compact) < 4:
        return None

    conversation_text = _format_messages_for_compaction(messages_to_compact)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": COMPACTOR_SYSTEM_PROMPT},
                {"role": "user", "content": conversation_text},
            ],
            max_tokens=max_output_tokens,
            temperature=0.0,
        )
        raw = response.choices[0].message.content or ""
        return _parse_compacted_output(raw)
    except Exception as e:
        print(f"  (Compaction failed: {e})")
        return None


def _format_messages_for_compaction(messages: list[dict]) -> str:
    """Format messages into a readable text block for the compactor LLM."""
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "") or ""

        if role == "tool":
            tool_name = msg.get("name", "unknown_tool")
            parts.append(f"[TOOL RESULT: {tool_name}]\n{content}")
        elif role == "assistant" and not content and msg.get("tool_calls"):
            tool_calls = msg["tool_calls"]
            calls_desc = []
            for tc in tool_calls:
                fn = tc.get("function", {})
                calls_desc.append(f"{fn.get('name', '?')}({fn.get('arguments', '')})")
            parts.append(f"ASSISTANT: [Called tools: {', '.join(calls_desc)}]")
        else:
            parts.append(f"{role.upper()}: {content}")

    return "\n\n".join(parts)


def _parse_compacted_output(raw: str) -> list[dict] | None:
    """Parse the LLM's JSON output into a list of message dicts."""
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:])
        if raw.endswith("```"):
            raw = raw[:-3].strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, list):
        return None

    result = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content", "")
        if role in ("user", "assistant") and content:
            result.append({"role": role, "content": content})

    return result if len(result) >= 2 else None
