"""
Context management for LLM conversations.

Handles:
- Token counting (local estimation via tiktoken + actual usage from API)
- Colored debug output for inspecting full context
"""

from __future__ import annotations

from dataclasses import dataclass, field

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

        try:
            self._enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._enc = None

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for a string using tiktoken."""
        if self._enc is None:
            return len(text) // 3
        return len(self._enc.encode(text))

    def estimate_messages_tokens(self, messages: list[dict] | None = None) -> int:
        """
        Estimate total tokens for a list of messages.

        Each message has ~4 token overhead (role, formatting).
        See: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        """
        if messages is None:
            messages = self.messages
        total = 3  # priming: <|start|>assistant<|message|>
        for msg in messages:
            total += 4
            total += self.estimate_tokens(msg.get("content", "") or "")
        return total

    def add_user_message(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})

    def pop_last_message(self) -> dict | None:
        """Remove and return the last message (for error recovery)."""
        if len(self.messages) > 1:
            return self.messages.pop()
        return None

    def record_usage(self, usage) -> TokenUsage:
        """Record API-reported token usage from a response."""
        token_usage = TokenUsage(
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage, "total_tokens", 0) or 0,
        )
        self.stats.record(token_usage)
        return token_usage

    def context_utilization(self) -> float:
        """Return current context utilization as a ratio (0.0 to 1.0+)."""
        return self.estimate_messages_tokens() / self.max_tokens

    def clear(self) -> None:
        """Reset conversation, keeping only the system prompt."""
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.stats = ConversationStats()

    # ── Display formatters ──────────────────────────────────────────────

    def format_stats(self) -> str:
        """Format conversation statistics for display."""
        s = self.stats
        estimated = self.estimate_messages_tokens()
        utilization = estimated / self.max_tokens * 100

        lines = [
            f"Messages:     {len(self.messages)} ({len(self.messages) - 1} excluding system)",
            f"Requests:     {s.total_requests}",
            f"",
            f"Cumulative API-reported usage:",
            f"  Prompt:     {s.total_prompt_tokens:,} tokens",
            f"  Completion: {s.total_completion_tokens:,} tokens",
            f"  Total:      {s.total_tokens:,} tokens",
            f"",
            f"Current context (estimated):",
            f"  Size:       ~{estimated:,} tokens",
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
            preview = content[:120].replace("\n", "\\n")
            if len(content) > 120:
                preview += "..."
            tokens = self.estimate_tokens(content)
            lines.append(f"  [{i:02d}] {role:10s} ~{tokens:>5d} tok | {preview}")
        return "\n".join(lines)

    def format_debug(self) -> str:
        """
        Format the full context with colored output for debugging.

        Shows exactly what would be sent to the LLM, with:
        - Color-coded roles (system=yellow, user=green, assistant=blue)
        - Token counts per message and running total
        - Visual separators between messages
        - Context utilization bar
        """
        C = Color
        estimated_total = self.estimate_messages_tokens()
        utilization = estimated_total / self.max_tokens * 100

        lines = []
        bar_width = 40
        filled = int(bar_width * min(utilization / 100, 1.0))
        bar_color = C.GREEN if utilization < 70 else (C.YELLOW if utilization < 90 else C.RED)
        bar = f"{bar_color}{'█' * filled}{C.DIM}{'░' * (bar_width - filled)}{C.RESET}"
        lines.append(
            f"{C.BOLD}Context Window{C.RESET}  "
            f"[{bar}]  "
            f"~{estimated_total:,} / {self.max_tokens:,} tokens ({utilization:.1f}%)"
        )
        lines.append(f"{C.DIM}{'─' * 80}{C.RESET}")

        running_tokens = 3
        for i, msg in enumerate(self.messages):
            role = msg["role"]
            content = msg.get("content", "") or ""
            msg_tokens = self.estimate_tokens(content) + 4
            running_tokens += msg_tokens

            role_color = C.role(role)
            header = (
                f"{role_color}{C.BOLD}[{i:02d}] {role.upper()}{C.RESET}"
                f"  {C.DIM}~{msg_tokens} tokens (cumulative: ~{running_tokens}){C.RESET}"
            )
            lines.append(header)

            for line in content.split("\n"):
                lines.append(f"  {role_color}{line}{C.RESET}")

            lines.append(f"{C.DIM}{'─' * 80}{C.RESET}")

        lines.append(
            f"{C.BOLD}Total:{C.RESET} {len(self.messages)} messages, "
            f"~{estimated_total:,} estimated tokens"
        )
        return "\n".join(lines)
