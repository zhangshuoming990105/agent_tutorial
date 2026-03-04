"""
Tool registry and built-in tools.

Provides a decorator-based registration system and a few demo tools.
Each tool is a plain Python function with a JSON schema describing its parameters.
"""

from __future__ import annotations

import json
import math
from typing import Any, Callable

# Global registry: name -> (function, schema)
_TOOL_REGISTRY: dict[str, tuple[Callable, dict]] = {}


def tool(name: str, description: str, parameters: dict):
    """
    Decorator to register a function as an LLM-callable tool.

    Usage:
        @tool("get_weather", "Get current weather", {"type": "object", ...})
        def get_weather(city: str) -> str:
            ...
    """

    def decorator(func: Callable) -> Callable:
        schema = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        }
        _TOOL_REGISTRY[name] = (func, schema)
        return func

    return decorator


def get_all_tool_schemas() -> list[dict]:
    """Return OpenAI-format tool schemas for all registered tools."""
    return [schema for _, schema in _TOOL_REGISTRY.values()]


def execute_tool(name: str, arguments: str) -> str:
    """
    Execute a registered tool by name with JSON-encoded arguments.
    Returns the result as a string, or an error message.
    """
    if name not in _TOOL_REGISTRY:
        return f"Error: unknown tool '{name}'"

    func, _ = _TOOL_REGISTRY[name]
    try:
        args = json.loads(arguments) if arguments else {}
        result = func(**args)
        return str(result)
    except json.JSONDecodeError as e:
        return f"Error: invalid JSON arguments: {e}"
    except TypeError as e:
        return f"Error: wrong arguments for tool '{name}': {e}"
    except Exception as e:
        return f"Error executing tool '{name}': {e}"


# ── Built-in demo tools ────────────────────────────────────────────────


@tool(
    name="calculator",
    description="Evaluate a mathematical expression. Supports +, -, *, /, **, sqrt, sin, cos, tan, log, pi, e.",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The math expression to evaluate, e.g. '2 ** 10' or 'sqrt(144)'",
            },
        },
        "required": ["expression"],
    },
)
def calculator(expression: str) -> str:
    safe_dict = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "abs": abs,
        "round": round,
        "pi": math.pi,
        "e": math.e,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Cannot evaluate '{expression}': {e}"


@tool(
    name="get_current_time",
    description="Get the current date and time.",
    parameters={
        "type": "object",
        "properties": {},
    },
)
def get_current_time() -> str:
    from datetime import datetime

    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S (%A)")
