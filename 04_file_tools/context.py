"""
Context management for LLM conversations.

Handles:
- Token counting (local estimation via tiktoken + actual usage from API)
- Colored debug output for inspecting full context
- Tool message tracking
- Smart context compaction (via compactor module)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json

import tiktoken


class Color:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    GRAY = "\033[90m"

    ROLE_COLORS = {
        "system": YELLOW,
        "user": GREEN,
        "assistant": BLUE,
        "tool": MAGENTA,
    }

    @classmethod
    def role(cls, role: str) -> str:
        return cls.ROLE_COLORS.get(role, cls.RESET)


@dataclass
class TokenUsage:
    """Token usage reported by the API for a single request."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ConversationStats:
    """Cumulative statistics for the conversation."""

    total_requests: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    history: list[TokenUsage] = field(default_factory=list)

    def record(self, usage: TokenUsage) -> None:
        self.total_requests += 1
        self.total_prompt_tokens += usage.prompt_tokens
        self.total_completion_tokens += usage.completion_tokens
        self.total_tokens += usage.total_tokens
        self.history.append(usage)


class ContextManager:
    """
    Manages conversation messages and context window budget.

    Uses tiktoken for local token estimation (cl100k_base as a reasonable
    cross-model approximation) and the API's reported usage for accurate tracking.
    """

    def __init__(self, system_prompt: str, max_tokens: int = 128_000):
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.messages: list[dict] = [{"role": "system", "content": system_prompt}]
        self.stats = ConversationStats()

        # Tracks last API prompt calibration:
        # prompt_tokens ~= managed_messages_local + request_overhead_local + hidden_provider_overhead
        self._last_prompt_tokens: int | None = None
        self._last_prompt_local_tokens: int = 0
        self._last_overhead_tokens: int = 0

        try:
            self._enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._enc = None

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for a string using tiktoken."""
        if self._enc is None:
            return len(text) // 3
        return len(self._enc.encode(text))

    def _estimate_message_tokens(
        self, msg: dict, include_metadata: bool = False
    ) -> int:
        """Estimate tokens for one message."""
        total = 4  # per-message framing overhead
        total += self.estimate_tokens(msg.get("content", "") or "")
        if include_metadata:
            meta = {k: v for k, v in msg.items() if k != "content"}
            if meta:
                total += self.estimate_tokens(json.dumps(meta, ensure_ascii=False))
        return total

    def estimate_messages_tokens(
        self, messages: list[dict] | None = None, include_metadata: bool = False
    ) -> int:
        """
        Estimate total tokens for a list of messages.

        Each message has ~4 token overhead (role, formatting).
        See: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        """
        if messages is None:
            messages = self.messages
        total = 3  # priming: <|start|>assistant<|message|>
        for msg in messages:
            total += self._estimate_message_tokens(msg, include_metadata=include_metadata)
        return total

    def estimate_messages_tokens_structured(self, messages: list[dict] | None = None) -> int:
        """
        Estimate tokens including structured metadata like tool_calls/tool ids/name.
        This is closer to provider prompt construction than content-only estimation.
        """
        return self.estimate_messages_tokens(messages, include_metadata=True)

    def add_user_message(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})

    def add_assistant_tool_calls(self, message) -> None:
        """Add an assistant message that contains tool calls (from API response)."""
        self.messages.append(message.model_dump())

    def add_tool_result(self, tool_call_id: str, name: str, content: str) -> None:
        """Add a tool result message."""
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": name,
                "content": content,
            }
        )

    def needs_compaction(
        self, buffer_ratio: float = 0.85, overhead_tokens: int | None = None
    ) -> bool:
        """Check if messages are approaching the context window limit."""
        return self.context_utilization(overhead_tokens=overhead_tokens) > buffer_ratio

    def apply_compacted_messages(
        self, compacted: list[dict], keep_recent: int = 6
    ) -> tuple[int, int]:
        """
        Replace old messages with compacted versions, keeping recent ones intact.

        Returns (num_old_messages_replaced, num_compacted_messages).
        """
        if len(self.messages) <= keep_recent + 1:
            return 0, 0

        system = [self.messages[0]]
        recent = self.messages[-keep_recent:]
        replaced_count = len(self.messages) - 1 - keep_recent

        self.messages = system + compacted + recent
        return replaced_count, len(compacted)

    def pop_last_message(self) -> dict | None:
        """Remove and return the last message (for error recovery)."""
        if len(self.messages) > 1:
            return self.messages.pop()
        return None

    def record_usage(self, usage, overhead_tokens: int = 0) -> TokenUsage:
        """Record API-reported token usage and update calibration baseline."""
        token_usage = TokenUsage(
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage, "total_tokens", 0) or 0,
        )
        self.stats.record(token_usage)

        self._last_prompt_tokens = token_usage.prompt_tokens
        self._last_prompt_local_tokens = self.estimate_messages_tokens_structured()
        self._last_overhead_tokens = max(0, overhead_tokens)

        return token_usage

    def get_context_tokens(self, overhead_tokens: int | None = None) -> int:
        """
        Best estimate of current context size in tokens.

        Uses API prompt calibration + local deltas:
        current ~= last_prompt_api
                   + (managed_local_now - managed_local_at_last_api)
                   + (overhead_now - overhead_at_last_api)
        """
        if overhead_tokens is None:
            overhead_tokens = self._last_overhead_tokens
        if self._last_prompt_tokens is not None:
            current_local = self.estimate_messages_tokens_structured()
            delta_local = current_local - self._last_prompt_local_tokens
            delta_overhead = max(0, overhead_tokens) - self._last_overhead_tokens
            return max(0, self._last_prompt_tokens + delta_local + delta_overhead)
        return self.estimate_messages_tokens_structured() + max(0, overhead_tokens)

    def get_token_diagnostics(
        self, schema_tokens_estimate: int = 0, skill_tokens_estimate: int = 0
    ) -> dict:
        """
        Return a diagnostics breakdown for current context token accounting.
        """
        overhead = max(0, schema_tokens_estimate) + max(0, skill_tokens_estimate)
        effective = self.get_context_tokens(overhead_tokens=overhead)
        content_only = self.estimate_messages_tokens()
        structured = self.estimate_messages_tokens_structured()
        hidden_overhead = max(0, effective - structured - overhead)
        return {
            "effective": effective,
            "content_only": content_only,
            "structured": structured,
            "schema_estimate": max(0, schema_tokens_estimate),
            "skill_estimate": max(0, skill_tokens_estimate),
            "overhead_estimate": overhead,
            "hidden_overhead_estimate": hidden_overhead,
        }

    def context_utilization(self, overhead_tokens: int | None = None) -> float:
        """Return current context utilization as a ratio (0.0 to 1.0+)."""
        return self.get_context_tokens(overhead_tokens=overhead_tokens) / self.max_tokens

    def clear(self) -> None:
        """Reset conversation, keeping only the system prompt."""
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.stats = ConversationStats()
        self._last_prompt_tokens = None
        self._last_prompt_local_tokens = 0
        self._last_overhead_tokens = 0

    def format_stats(self) -> str:
        """Format conversation statistics for display."""
        s = self.stats
        calibrated = self.get_context_tokens()
        structured = self.estimate_messages_tokens_structured()
        content_only = self.estimate_messages_tokens()
        utilization = calibrated / self.max_tokens * 100

        lines = [
            f"Messages:     {len(self.messages)} ({len(self.messages) - 1} excluding system)",
            f"Requests:     {s.total_requests}",
            "",
            "Cumulative API-reported usage:",
            f"  Prompt:     {s.total_prompt_tokens:,} tokens",
            f"  Completion: {s.total_completion_tokens:,} tokens",
            f"  Total:      {s.total_tokens:,} tokens",
            "",
            "Current context:",
            f"  Size:       ~{calibrated:,} tokens"
            + (
                "  (calibrated from API)"
                if self._last_prompt_tokens
                else "  (local estimate)"
            ),
            f"  Structured: ~{structured:,} tokens (local, with metadata)",
            f"  Content:    ~{content_only:,} tokens (local, content only)",
            f"  Limit:      {self.max_tokens:,} tokens",
            f"  Usage:      {utilization:.1f}%",
        ]
        return "\n".join(lines)

    def format_history(self) -> str:
        """Format conversation messages for display (compact, one line each)."""
        lines = []
        for i, msg in enumerate(self.messages):
            role = msg["role"].upper()
            content = msg.get("content", "") or ""

            if role == "TOOL":
                name = msg.get("name", "?")
                preview = content[:80].replace("\n", "\\n")
                if len(content) > 80:
                    preview += "..."
                tokens = self._estimate_message_tokens(msg, include_metadata=True)
                lines.append(
                    f"  [{i:02d}] {role:10s} ~{tokens:>5d} tok | [{name}] {preview}"
                )
            elif role == "ASSISTANT" and not content and msg.get("tool_calls"):
                tool_calls = msg["tool_calls"]
                names = [tc.get("function", {}).get("name", "?") for tc in tool_calls]
                tokens = self._estimate_message_tokens(msg, include_metadata=True)
                lines.append(f"  [{i:02d}] {role:10s} ~{tokens:>5d} tok | -> called: {', '.join(names)}")
            else:
                preview = content[:120].replace("\n", "\\n")
                if len(content) > 120:
                    preview += "..."
                tokens = self._estimate_message_tokens(msg, include_metadata=True)
                lines.append(f"  [{i:02d}] {role:10s} ~{tokens:>5d} tok | {preview}")
        return "\n".join(lines)

    def format_debug(self) -> str:
        """Format the full context with colored output for debugging."""
        c = Color
        calibrated_total = self.get_context_tokens()
        utilization = calibrated_total / self.max_tokens * 100

        lines = []
        bar_width = 40
        filled = int(bar_width * min(utilization / 100, 1.0))
        bar_color = c.GREEN if utilization < 70 else (c.YELLOW if utilization < 90 else c.RED)
        bar = f"{bar_color}{'█' * filled}{c.DIM}{'░' * (bar_width - filled)}{c.RESET}"
        source = "calibrated" if self._last_prompt_tokens else "estimated"
        lines.append(
            f"{c.BOLD}Context Window{c.RESET}  "
            f"[{bar}]  "
            f"~{calibrated_total:,} / {self.max_tokens:,} tokens ({utilization:.1f}%) [{source}]"
        )
        lines.append(f"{c.DIM}{'-' * 80}{c.RESET}")

        running_tokens = 3
        for i, msg in enumerate(self.messages):
            role = msg["role"]
            content = msg.get("content", "") or ""
            msg_tokens = self._estimate_message_tokens(msg, include_metadata=True)
            running_tokens += msg_tokens
            role_color = c.role(role)

            if role == "tool":
                name = msg.get("name", "?")
                header = (
                    f"{role_color}{c.BOLD}[{i:02d}] TOOL ({name}){c.RESET}"
                    f"  {c.DIM}~{msg_tokens} tokens (cumulative: ~{running_tokens}){c.RESET}"
                )
            elif role == "assistant" and not content and msg.get("tool_calls"):
                tool_calls = msg["tool_calls"]
                names = [tc.get("function", {}).get("name", "?") for tc in tool_calls]
                header = (
                    f"{role_color}{c.BOLD}[{i:02d}] ASSISTANT -> {', '.join(names)}{c.RESET}"
                    f"  {c.DIM}(cumulative: ~{running_tokens}){c.RESET}"
                )
            else:
                header = (
                    f"{role_color}{c.BOLD}[{i:02d}] {role.upper()}{c.RESET}"
                    f"  {c.DIM}~{msg_tokens} tokens (cumulative: ~{running_tokens}){c.RESET}"
                )
            lines.append(header)

            if role == "assistant" and not content and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    fn = tc.get("function", {})
                    args = fn.get("arguments", "")
                    lines.append(f"  {role_color}{fn.get('name', '?')}({args}){c.RESET}")
            else:
                for line in content.split("\n"):
                    lines.append(f"  {role_color}{line}{c.RESET}")

            lines.append(f"{c.DIM}{'-' * 80}{c.RESET}")

        lines.append(
            f"{c.BOLD}Total:{c.RESET} {len(self.messages)} messages, "
            f"~{calibrated_total:,} tokens [{source}]"
        )
        return "\n".join(lines)
