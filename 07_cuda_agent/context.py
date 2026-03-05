"""
Context management for LLM conversations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re

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
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ConversationStats:
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
    def __init__(self, system_prompt: str, max_tokens: int = 128_000):
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.messages: list[dict] = [{"role": "system", "content": system_prompt}]
        self.stats = ConversationStats()

        # API calibration state:
        # prompt_tokens ~= managed_messages_local + request_overhead_local + hidden_provider_overhead
        self._last_prompt_tokens: int | None = None
        self._last_prompt_local_tokens: int = 0
        self._last_overhead_tokens: int = 0

        try:
            self._enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._enc = None

    def estimate_tokens(self, text: str) -> int:
        if self._enc is None:
            return len(text) // 3
        return len(self._enc.encode(text))

    def _estimate_message_tokens(self, msg: dict, include_metadata: bool = False) -> int:
        total = 4
        total += self.estimate_tokens(msg.get("content", "") or "")
        if include_metadata:
            meta = {k: v for k, v in msg.items() if k != "content"}
            if meta:
                total += self.estimate_tokens(json.dumps(meta, ensure_ascii=False))
        return total

    def estimate_messages_tokens(
        self, messages: list[dict] | None = None, include_metadata: bool = False
    ) -> int:
        if messages is None:
            messages = self.messages
        total = 3
        for msg in messages:
            total += self._estimate_message_tokens(msg, include_metadata=include_metadata)
        return total

    def estimate_messages_tokens_structured(self, messages: list[dict] | None = None) -> int:
        return self.estimate_messages_tokens(messages, include_metadata=True)

    def add_user_message(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})

    def add_assistant_tool_calls(self, message) -> None:
        self.messages.append(message.model_dump())

    def add_tool_result(self, tool_call_id: str, name: str, content: str) -> None:
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
        return self.context_utilization(overhead_tokens=overhead_tokens) > buffer_ratio

    def apply_compacted_messages(
        self, compacted: list[dict], keep_recent: int = 6
    ) -> tuple[int, int]:
        if len(self.messages) <= keep_recent + 1:
            return 0, 0
        system = [self.messages[0]]
        recent = self.messages[-keep_recent:]
        replaced_count = len(self.messages) - 1 - keep_recent
        self.messages = system + compacted + recent
        return replaced_count, len(compacted)

    def pop_last_message(self) -> dict | None:
        if len(self.messages) > 1:
            return self.messages.pop()
        return None

    @staticmethod
    def _looks_like_failed_tool_result(content: str) -> bool:
        text = (content or "").strip()
        if not text:
            return False
        if text.startswith("Error") or text.startswith("Denied"):
            return True
        match = re.search(r"^exit_code=(-?\d+)$", text, flags=re.MULTILINE)
        if match:
            try:
                return int(match.group(1)) != 0
            except ValueError:
                return True
        return False

    def drop_failed_tool_messages(self, start_index: int = 1) -> int:
        """
        Remove failed tool-result traces from context to keep successful task history concise.

        Returns number of removed messages.
        """
        start_index = max(1, min(start_index, len(self.messages)))
        failed_tool_call_ids: set[str] = set()
        remove_indices: set[int] = set()

        for i in range(start_index, len(self.messages)):
            msg = self.messages[i]
            if msg.get("role") != "tool":
                continue
            if self._looks_like_failed_tool_result(msg.get("content", "") or ""):
                remove_indices.add(i)
                tcid = str(msg.get("tool_call_id", "") or "")
                if tcid:
                    failed_tool_call_ids.add(tcid)

        if not remove_indices:
            return 0

        # For assistant tool-call messages, drop only failed calls; drop the whole message if none remains.
        for i in range(start_index, len(self.messages)):
            if i in remove_indices:
                continue
            msg = self.messages[i]
            if msg.get("role") != "assistant":
                continue
            tool_calls = msg.get("tool_calls")
            if not isinstance(tool_calls, list) or not tool_calls:
                continue
            kept_calls = [
                tc for tc in tool_calls if str(tc.get("id", "") or "") not in failed_tool_call_ids
            ]
            if not kept_calls:
                remove_indices.add(i)
                continue
            if len(kept_calls) != len(tool_calls):
                msg["tool_calls"] = kept_calls

        original_len = len(self.messages)
        self.messages = [
            m for idx, m in enumerate(self.messages) if idx not in remove_indices
        ]
        return original_len - len(self.messages)

    def record_usage(self, usage, overhead_tokens: int = 0) -> TokenUsage:
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
        return self.get_context_tokens(overhead_tokens=overhead_tokens) / self.max_tokens

    def clear(self) -> None:
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.stats = ConversationStats()
        self._last_prompt_tokens = None
        self._last_prompt_local_tokens = 0
        self._last_overhead_tokens = 0

    def format_history(self) -> str:
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
                lines.append(
                    f"  [{i:02d}] {role:10s} ~{tokens:>5d} tok | -> called: {', '.join(names)}"
                )
            else:
                preview = content[:120].replace("\n", "\\n")
                if len(content) > 120:
                    preview += "..."
                tokens = self._estimate_message_tokens(msg, include_metadata=True)
                lines.append(f"  [{i:02d}] {role:10s} ~{tokens:>5d} tok | {preview}")
        return "\n".join(lines)

    def format_debug(self) -> str:
        c = Color
        total_tokens = self.get_context_tokens()
        utilization = total_tokens / self.max_tokens * 100
        source = "calibrated" if self._last_prompt_tokens else "estimated"

        lines = []
        bar_width = 40
        filled = int(bar_width * min(utilization / 100, 1.0))
        bar_color = c.GREEN if utilization < 70 else (c.YELLOW if utilization < 90 else c.RED)
        bar = f"{bar_color}{'█' * filled}{c.DIM}{'░' * (bar_width - filled)}{c.RESET}"
        lines.append(
            f"{c.BOLD}Context Window{c.RESET}  "
            f"[{bar}]  "
            f"~{total_tokens:,} / {self.max_tokens:,} tokens ({utilization:.1f}%) [{source}]"
        )
        lines.append(f"{c.DIM}{'-' * 80}{c.RESET}")

        running = 3
        for i, msg in enumerate(self.messages):
            role = msg["role"]
            content = msg.get("content", "") or ""
            msg_tokens = self._estimate_message_tokens(msg, include_metadata=True)
            running += msg_tokens
            role_color = c.role(role)

            if role == "tool":
                name = msg.get("name", "?")
                header = (
                    f"{role_color}{c.BOLD}[{i:02d}] TOOL ({name}){c.RESET}"
                    f"  {c.DIM}~{msg_tokens} tokens (cumulative: ~{running}){c.RESET}"
                )
            elif role == "assistant" and not content and msg.get("tool_calls"):
                names = [tc.get("function", {}).get("name", "?") for tc in msg["tool_calls"]]
                header = (
                    f"{role_color}{c.BOLD}[{i:02d}] ASSISTANT -> {', '.join(names)}{c.RESET}"
                    f"  {c.DIM}(cumulative: ~{running}){c.RESET}"
                )
            else:
                header = (
                    f"{role_color}{c.BOLD}[{i:02d}] {role.upper()}{c.RESET}"
                    f"  {c.DIM}~{msg_tokens} tokens (cumulative: ~{running}){c.RESET}"
                )
            lines.append(header)

            if role == "assistant" and not content and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    fn = tc.get("function", {})
                    lines.append(f"  {role_color}{fn.get('name', '?')}({fn.get('arguments', '')}){c.RESET}")
            else:
                for line in content.split("\n"):
                    lines.append(f"  {role_color}{line}{c.RESET}")

            lines.append(f"{c.DIM}{'-' * 80}{c.RESET}")

        lines.append(
            f"{c.BOLD}Total:{c.RESET} {len(self.messages)} messages, ~{total_tokens:,} tokens [{source}]"
        )
        return "\n".join(lines)
