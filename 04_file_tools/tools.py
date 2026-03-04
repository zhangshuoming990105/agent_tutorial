"""
Tool registry and built-in tools for Step 04.

This step extends generic tools (calculator/time) with filesystem tools:
- list_directory
- read_file
- write_file
- search_files (ripgrep/regex)
- grep_text (literal grep-style search)
"""

from __future__ import annotations

import json
import math
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Callable

# Global registry: name -> (function, schema)
_TOOL_REGISTRY: dict[str, tuple[Callable, dict]] = {}


def _workspace_root() -> Path:
    # Default workspace root: parent of this step folder (agent_tutorial/)
    return Path(__file__).resolve().parents[1]


def _resolve_in_workspace(path_str: str) -> Path:
    root = _workspace_root()
    raw = Path(path_str)
    target = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    try:
        target.relative_to(root)
    except ValueError as e:
        raise ValueError(f"path '{path_str}' is outside workspace root '{root}'") from e
    return target


def _rel(path: Path) -> str:
    return str(path.relative_to(_workspace_root()))


def tool(name: str, description: str, parameters: dict):
    """Decorator to register a function as an LLM-callable tool."""

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


def get_tool_schema_map() -> dict[str, dict]:
    """Return a map: tool_name -> OpenAI-format tool schema."""
    return {name: schema for name, (_, schema) in _TOOL_REGISTRY.items()}


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


@tool(
    name="workspace_info",
    description="Return workspace root information for path-aware tool usage.",
    parameters={"type": "object", "properties": {}},
)
def workspace_info() -> str:
    root = _workspace_root()
    return f"workspace_root={root}"


@tool(
    name="list_directory",
    description="List files and folders in a directory under workspace root.",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory path relative to workspace root, e.g. '.' or '03_tool_use'",
            },
            "include_hidden": {
                "type": "boolean",
                "description": "Whether to include hidden files/directories.",
            },
            "max_entries": {
                "type": "integer",
                "description": "Maximum entries to return (1-500).",
            },
        },
    },
)
def list_directory(
    path: str = ".",
    include_hidden: bool = False,
    max_entries: int = 200,
) -> str:
    max_entries = max(1, min(max_entries, 500))
    target = _resolve_in_workspace(path)
    if not target.exists():
        return f"Error: directory not found: {_rel(target)}"
    if not target.is_dir():
        return f"Error: not a directory: {_rel(target)}"

    entries = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    if not include_hidden:
        entries = [e for e in entries if not e.name.startswith(".")]

    lines = [f"Directory: {_rel(target)}"]
    for entry in entries[:max_entries]:
        kind = "dir " if entry.is_dir() else "file"
        size = "-" if entry.is_dir() else str(entry.stat().st_size)
        lines.append(f"{kind:4}  {size:>8}  {_rel(entry)}")

    if len(entries) > max_entries:
        lines.append(f"... ({len(entries) - max_entries} more entries)")

    return "\n".join(lines)


@tool(
    name="read_file",
    description="Read text file content with line numbers.",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path relative to workspace root.",
            },
            "start_line": {
                "type": "integer",
                "description": "1-indexed start line.",
            },
            "max_lines": {
                "type": "integer",
                "description": "Maximum number of lines to read (1-500).",
            },
        },
        "required": ["path"],
    },
)
def read_file(path: str, start_line: int = 1, max_lines: int = 200) -> str:
    target = _resolve_in_workspace(path)
    if not target.exists():
        return f"Error: file not found: {_rel(target)}"
    if not target.is_file():
        return f"Error: not a file: {_rel(target)}"

    max_lines = max(1, min(max_lines, 500))
    start_line = max(1, start_line)

    try:
        content = target.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error reading file '{_rel(target)}': {e}"

    lines = content.splitlines()
    if not lines:
        return f"File is empty: {_rel(target)}"

    start_idx = start_line - 1
    if start_idx >= len(lines):
        return f"Error: start_line {start_line} exceeds file length {len(lines)}"

    end_idx = min(start_idx + max_lines, len(lines))
    out = [f"File: {_rel(target)} (lines {start_line}-{end_idx} of {len(lines)})"]
    for i in range(start_idx, end_idx):
        out.append(f"{i + 1}|{lines[i]}")
    if end_idx < len(lines):
        out.append(f"... ({len(lines) - end_idx} more lines)")
    return "\n".join(out)


@tool(
    name="write_file",
    description="Write text content to a file under workspace root (overwrite or append).",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path relative to workspace root.",
            },
            "content": {
                "type": "string",
                "description": "Text content to write.",
            },
            "mode": {
                "type": "string",
                "enum": ["overwrite", "append"],
                "description": "Write mode.",
            },
        },
        "required": ["path", "content"],
    },
)
def write_file(path: str, content: str, mode: str = "overwrite") -> str:
    target = _resolve_in_workspace(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if mode not in ("overwrite", "append"):
        return "Error: mode must be 'overwrite' or 'append'"

    try:
        if mode == "overwrite":
            target.write_text(content, encoding="utf-8")
        else:
            with target.open("a", encoding="utf-8") as f:
                f.write(content)
        return (
            f"Wrote {len(content)} chars to {_rel(target)} (mode={mode}) "
            f"at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
    except Exception as e:
        return f"Error writing file '{_rel(target)}': {e}"


@tool(
    name="search_files",
    description="Search text in files using ripgrep and return matching lines.",
    parameters={
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Regex pattern to search for.",
            },
            "path": {
                "type": "string",
                "description": "Directory or file path relative to workspace root.",
            },
            "glob": {
                "type": "string",
                "description": "Optional file glob, e.g. '*.py' or '*.{ts,tsx}'.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of matching lines to return (1-200).",
            },
            "case_sensitive": {
                "type": "boolean",
                "description": "Whether the search should be case sensitive.",
            },
        },
        "required": ["pattern"],
    },
)
def search_files(
    pattern: str,
    path: str = ".",
    glob: str | None = None,
    max_results: int = 50,
    case_sensitive: bool = True,
) -> str:
    target = _resolve_in_workspace(path)
    max_results = max(1, min(max_results, 200))

    cmd = ["rg", "-n", pattern, str(target)]
    if glob:
        cmd.extend(["--glob", glob])
    if not case_sensitive:
        cmd.append("-i")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return _search_files_python(
            pattern=pattern,
            target=target,
            glob=glob,
            max_results=max_results,
            case_sensitive=case_sensitive,
        )
    except Exception as e:
        return f"Error running search: {e}"

    if result.returncode not in (0, 1):
        return f"Error searching files: {result.stderr.strip() or result.stdout.strip()}"

    matches = [line for line in result.stdout.splitlines() if line.strip()]
    if not matches:
        return "No matches found."

    out = [f"Search root: {_rel(target) if target.exists() else path}", f"Pattern: {pattern}"]
    for line in matches[:max_results]:
        out.append(line)
    if len(matches) > max_results:
        out.append(f"... ({len(matches) - max_results} more matches)")
    return "\n".join(out)


@tool(
    name="grep_text",
    description="Grep-style text search across files. By default treats pattern as plain text (not regex).",
    parameters={
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Text (or regex when regex=true) to search for.",
            },
            "path": {
                "type": "string",
                "description": "Directory or file path relative to workspace root.",
            },
            "glob": {
                "type": "string",
                "description": "Optional file glob, e.g. '*.py' or '*.{ts,tsx}'.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of matching lines to return (1-200).",
            },
            "case_sensitive": {
                "type": "boolean",
                "description": "Whether search should be case sensitive.",
            },
            "regex": {
                "type": "boolean",
                "description": "If true, pattern is treated as regex; otherwise literal text.",
            },
        },
        "required": ["pattern"],
    },
)
def grep_text(
    pattern: str,
    path: str = ".",
    glob: str | None = None,
    max_results: int = 50,
    case_sensitive: bool = True,
    regex: bool = False,
) -> str:
    query = pattern if regex else re.escape(pattern)
    return search_files(
        pattern=query,
        path=path,
        glob=glob,
        max_results=max_results,
        case_sensitive=case_sensitive,
    )


def _search_files_python(
    pattern: str,
    target: Path,
    glob: str | None,
    max_results: int,
    case_sensitive: bool,
) -> str:
    """Fallback search implementation when rg is unavailable."""
    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        regex = re.compile(pattern, flags)
    except re.error as e:
        return f"Error: invalid regex pattern '{pattern}': {e}"

    if target.is_file():
        candidates = [target] if _is_searchable_text_file(target) else []
    else:
        if glob:
            candidates = [p for p in target.rglob(glob) if p.is_file()]
        else:
            candidates = [p for p in target.rglob("*") if p.is_file()]
        candidates = [p for p in candidates if _is_searchable_text_file(p)]

    matches: list[str] = []
    for file_path in candidates:
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        for i, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                matches.append(f"{_rel(file_path)}:{i}:{line}")
                if len(matches) >= max_results:
                    break
        if len(matches) >= max_results:
            break

    if not matches:
        return "No matches found."

    out = [
        "Search backend: python-fallback (rg unavailable)",
        f"Search root: {_rel(target) if target.exists() else str(target)}",
        f"Pattern: {pattern}",
    ]
    out.extend(matches)
    return "\n".join(out)


def _is_searchable_text_file(path: Path) -> bool:
    """Heuristic filter to skip binary/cache files for fallback search."""
    if "__pycache__" in path.parts:
        return False
    if path.suffix.lower() in {".pyc", ".pyo", ".so", ".dylib", ".exe", ".bin"}:
        return False
    try:
        with path.open("rb") as f:
            sample = f.read(2048)
        if b"\x00" in sample:
            return False
    except Exception:
        return False
    return True


# Keep generic demo tools from Step 03.


@tool(
    name="calculator",
    description="Evaluate a mathematical expression. Supports +, -, *, /, **, sqrt, sin, cos, tan, log, pi, e.",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The math expression to evaluate, e.g. '2 ** 10' or 'sqrt(144)'",
            }
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
    parameters={"type": "object", "properties": {}},
)
def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S (%A)")
