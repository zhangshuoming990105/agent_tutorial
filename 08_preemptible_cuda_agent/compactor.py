"""
Smart context compaction for old turns only.

Two compression levels:
  low  - conservative: summarise only redundant/verbose parts, keep structure and key detail.
  high - aggressive: collapse everything into minimal summary pairs, keep only essential facts.
"""

from __future__ import annotations

import json

from openai import OpenAI

# ---------------------------------------------------------------------------
# System prompts for each compression level
# ---------------------------------------------------------------------------

COMPACTOR_SYSTEM_PROMPT_LOW = """\
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

COMPACTOR_SYSTEM_PROMPT_HIGH = """\
You are an aggressive conversation compressor. Given older conversation history, produce the \
most compact possible summary that still lets future turns proceed correctly.

Rules:
1. Output MUST be a valid JSON array of messages: [{"role":"user","content":"..."},{"role":"assistant","content":"..."}].
2. Collapse the entire history into as few user/assistant pairs as possible (ideally 1–3 pairs).
3. For each collapsed pair:
   - user: one short sentence capturing the overall goal or key request.
   - assistant: a tight bullet-list of outcomes, file paths created/modified, commands run, and key results. No prose.
4. Preserve ONLY hard facts: exact file paths, function/symbol names, shell commands, numeric results, error messages.
5. Discard everything else: reasoning, intermediate steps, retried attempts, verbose tool outputs.
6. Write in the same language as source conversation.
7. Output ONLY JSON array, no markdown, no commentary.
"""

# Backward-compatible alias (old code may import COMPACTOR_SYSTEM_PROMPT)
COMPACTOR_SYSTEM_PROMPT = COMPACTOR_SYSTEM_PROMPT_LOW

# Output token budgets per level.
# "high" uses a larger budget than might be expected: aggressive compression still
# needs enough tokens to produce valid JSON for potentially many message groups.
# The aggressiveness comes from the prompt, not artificial output starvation.
_MAX_OUTPUT_TOKENS: dict[str, int] = {
    "low": 2400,
    "high": 2000,
}

# When the input is very large (many messages), scale the budget proportionally.
_TOKENS_PER_INPUT_MESSAGE = 30   # extra tokens per message beyond a base count
_BASE_MESSAGE_THRESHOLD = 20     # below this count, use the flat budget above


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compact_messages(
    client: OpenAI,
    model: str,
    messages_to_compact: list[dict],
    level: str = "low",
    max_output_tokens: int | None = None,
) -> list[dict] | None:
    """Compress *messages_to_compact* using the LLM.

    Args:
        level: "low" (conservative) or "high" (aggressive).
        max_output_tokens: override token budget; if None uses level default.
    """
    if len(messages_to_compact) < 4:
        return None

    level = level if level in ("low", "high") else "low"
    system_prompt = (
        COMPACTOR_SYSTEM_PROMPT_HIGH if level == "high" else COMPACTOR_SYSTEM_PROMPT_LOW
    )

    # Scale output budget with input size so large histories aren't truncated.
    if max_output_tokens is not None:
        budget = max_output_tokens
    else:
        base = _MAX_OUTPUT_TOKENS[level]
        n = len(messages_to_compact)
        if n > _BASE_MESSAGE_THRESHOLD:
            budget = base + (n - _BASE_MESSAGE_THRESHOLD) * _TOKENS_PER_INPUT_MESSAGE
        else:
            budget = base

    conversation_text = _format_messages_for_compaction(messages_to_compact)

    def _call_once() -> list[dict] | None:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": conversation_text},
            ],
            max_tokens=budget,
            temperature=0.0,
        )
        choice = response.choices[0]
        raw = choice.message.content or ""
        finish = choice.finish_reason

        # If the model hit the token limit, try to salvage truncated JSON.
        if finish == "length" and raw:
            raw = _repair_truncated_json(raw)

        result = _parse_compacted_output(raw)
        if result is None:
            out_tok = getattr(response.usage, "completion_tokens", "?")
            print(f"  (Compaction parse failed. finish_reason={finish}, "
                  f"output_tokens={out_tok}, budget={budget}, "
                  f"raw_preview={repr(raw[:300])})")
        return result

    import time as _time
    max_retries = 2
    for attempt in range(max_retries):
        try:
            return _call_once()
        except Exception as e:
            err_str = str(e)
            is_rate_limit = "429" in err_str or "rate limit" in err_str.lower()
            if is_rate_limit and attempt < max_retries - 1:
                wait = 10 * (attempt + 1)
                print(f"  (Compaction rate-limited, retrying in {wait}s... attempt {attempt+1}/{max_retries})")
                _time.sleep(wait)
                continue
            print(f"  (Compaction failed: {e})")
            return None
    return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _repair_truncated_json(raw: str) -> str:
    """Best-effort repair of a truncated JSON array from a compactor response.

    Strips markdown fences, then tries to close the array after the last
    complete object so json.loads can succeed.
    """
    s = raw.strip()
    # Strip markdown fences
    if s.startswith("```"):
        lines = s.split("\n")
        s = "\n".join(lines[1:]).rstrip("`").strip()

    # Find the last '}' that closes a complete object
    last_brace = s.rfind("}")
    if last_brace == -1:
        return raw  # nothing to repair
    candidate = s[: last_brace + 1]
    # Remove a trailing comma if present
    candidate = candidate.rstrip().rstrip(",")
    # Ensure it's wrapped as an array
    if not candidate.lstrip().startswith("["):
        candidate = "[" + candidate
    candidate = candidate + "]"
    return candidate


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
