"""
Shared runtime state for preemption across chatbot and tools.

This module is intentionally tiny to avoid import cycles.
"""

from __future__ import annotations

import threading

_lock = threading.Lock()
_autonomous_turn = False
_shell_interrupt_on_preempt = False
_preempt_requested = threading.Event()


def set_autonomous_turn(active: bool) -> None:
    global _autonomous_turn
    with _lock:
        _autonomous_turn = bool(active)


def is_autonomous_turn() -> bool:
    with _lock:
        return _autonomous_turn


def request_preempt() -> None:
    _preempt_requested.set()


def clear_preempt_request() -> None:
    _preempt_requested.clear()


def is_preempt_requested() -> bool:
    return _preempt_requested.is_set()


def set_shell_interrupt_on_preempt(enabled: bool) -> None:
    global _shell_interrupt_on_preempt
    with _lock:
        _shell_interrupt_on_preempt = bool(enabled)


def shell_interrupt_on_preempt() -> bool:
    with _lock:
        return _shell_interrupt_on_preempt
