"""
Smart context compaction for old turns only.
"""

from __future__ import annotations

import json

from openai import OpenAI

COMPACTOR_SYSTEM_PROMPT = """\
You are a conversation compressor. Given older conversation history, produce a \
shorter version that preserves essential information for future turns.

Rules:
1. Output MUST be a valid JSON array of messages: [{"role":"user","content":"..."},{"role":"assistant","content":"..."}].
2. Keep multi-turn structure as user/assistant pairs; do not output one big paragraph.
3. For user messages: keep short prompts verbatim; summarize long prompts to intent + constraints.
4. For assistant messages: keep deliverables only (final answer, code, commands, file paths, key facts).
5. Preserve critical exact fields when present: file paths, symbols, commands, versions, numeric results.
6. Remove chain-of-thought style reasoning, retries, failed attempts, and repetitive explanations.
7. For tool sequences, convert to concise result-oriented assistant messages.
8. Merge adjacent messages of same role when it improves clarity.
9. These are OLD turns only. Compress old code/tool outputs aggressively:
   - replace long code blocks with short summaries (path + key symbols + outcome),
   - keep exact snippets only when required for correctness.
10. Assume recent turns are preserved elsewhere; avoid repeating recent details.
11. Write in the same language as source conversation.
12. Output ONLY JSON array, no markdown, no commentary.
"""


def compact_messages(
    client: OpenAI,
    model: str,
    messages_to_compact: list[dict],
    max_output_tokens: int = 1200,
) -> list[dict] | None:
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
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "") or ""

        if role == "tool":
            tool_name = msg.get("name", "unknown_tool")
            parts.append(f"[TOOL RESULT: {tool_name}]\n{content}")
        elif role == "assistant" and not content and msg.get("tool_calls"):
            calls_desc = []
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                calls_desc.append(f"{fn.get('name', '?')}({fn.get('arguments', '')})")
            parts.append(f"ASSISTANT: [Called tools: {', '.join(calls_desc)}]")
        else:
            parts.append(f"{role.upper()}: {content}")

    return "\n\n".join(parts)


def _parse_compacted_output(raw: str) -> list[dict] | None:
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
