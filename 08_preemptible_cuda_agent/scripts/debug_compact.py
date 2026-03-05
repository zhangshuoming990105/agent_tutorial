#!/usr/bin/env python3
"""
Debug script for /compact: minimal context with tool calls.
Run: cd 08_preemptible_cuda_agent && python scripts/debug_compact.py
Requires: KSYUN_API_KEY or INFINI_API_KEY
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from llm import create_client

from compactor import (
    COMPACTOR_SYSTEM_PROMPT,
    _format_messages_for_compaction,
    _parse_compacted_output,
    compact_messages,
)


def make_minimal_messages_with_tools() -> list[dict]:
    """Minimal conversation: user -> tool call -> tool result -> assistant (x2 rounds)."""
    return [
        {"role": "user", "content": "现在几点了？"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {"name": "get_current_time", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "get_current_time",
            "content": "2026-03-06 01:02:05 (Friday)",
        },
        {"role": "assistant", "content": "当前时间是 2026-03-06 01:02:05 (Friday)。"},
        {"role": "user", "content": "在 temp/debug 下写一个 hello.txt，内容写 Hello World"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_2",
                    "function": {
                        "name": "write_file",
                        "arguments": json.dumps(
                            {"path": "temp/debug/hello.txt", "content": "Hello World"}
                        ),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_2",
            "name": "write_file",
            "content": "Wrote 11 chars to temp/debug/hello.txt (mode=overwrite) at 2026-03-06 01:02:10",
        },
        {"role": "assistant", "content": "已写入 temp/debug/hello.txt。"},
    ]


def make_large_messages_with_tools(count_rounds: int = 6) -> list[dict]:
    """Repeat tool-call rounds to reach ~48 messages (8 per round)."""
    base = make_minimal_messages_with_tools()
    out = []
    for i in range(count_rounds):
        for m in base:
            msg = {k: v for k, v in m.items()}  # shallow copy
            if msg["role"] == "tool" and msg.get("name") == "get_current_time":
                msg = dict(msg, content=f"2026-03-06 01:{i:02d}:00 (Friday)")
            out.append(msg)
    return out


def make_realistic_messages_with_long_tool_output() -> list[dict]:
    """Simulate real session: long read_file / run_shell outputs like agent_test."""
    long_c_content = """#include <stdio.h>

int main() {
    printf("Hello from C!\\n");
    return 0;
}
"""
    long_py_content = '''"""Fibonacci sequence generator."""
def fibonacci(n):
    a, b = 0, 1
    result = []
    for _ in range(n):
        result.append(a)
        a, b = b, a + b
    return result
if __name__ == "__main__":
    fib = fibonacci(15)
    print(f"First 15 Fibonacci numbers: {fib}")
'''
    # Simulate full sorting.py (26 lines) - like real agent_test
    sorting_py = '''"""Sorting algorithm demo: bubble sort vs built-in sort."""
import time
import random
def bubble_sort(arr):
    arr = arr[:]
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
if __name__ == "__main__":
    data = [random.randint(1, 1000) for _ in range(500)]
    t0 = time.perf_counter()
    result_bubble = bubble_sort(data)
    t1 = time.perf_counter()
    result_builtin = sorted(data)
    t2 = time.perf_counter()
    assert result_bubble == result_builtin, "Results mismatch!"
    print(f"Bubble sort:  {(t1-t0)*1000:.3f} ms")
    print(f"Built-in sort: {(t2-t1)*1000:.3f} ms")
'''
    run_shell_output = """exit_code=0
cwd=/workspace/...
command=./temp/20260306_010205/factorial
stdout:
 0! = 1
 1! = 1
 2! = 2
 3! = 6
 4! = 24
 5! = 120
 6! = 720
 7! = 5040
 8! = 40320
 9! = 362880
10! = 3628800
"""
    return [
        {"role": "user", "content": "在 temp 下创建文件夹并写 C 和 Python 程序"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "function": {"name": "run_shell", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "run_shell", "content": "exit_code=0\nmkdir temp/20260306_010205"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c2", "function": {"name": "write_file", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c2", "name": "write_file", "content": f"Wrote {len(long_c_content)} chars to temp/.../hello.c"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c3", "function": {"name": "write_file", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c3", "name": "write_file", "content": f"Wrote {len(long_py_content)} chars to temp/.../fibonacci.py"},
        {"role": "assistant", "content": "已写入 hello.c 和 fibonacci.py。"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c4", "function": {"name": "read_file", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c4", "name": "read_file", "content": f"File: hello.c (lines 1-6 of 6)\n1|{long_c_content}\n\nFile: factorial.c (lines 1-13 of 13)\n1|#include <stdio.h>\n3|long long factorial(int n) {{\n...\n13|}}\n\nFile: sorting.py (lines 1-26 of 26)\n1|{sorting_py}"},
        {"role": "assistant", "content": "已读取并确认内容正确。"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c5", "function": {"name": "run_shell", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c5", "name": "run_shell", "content": run_shell_output},
        {"role": "assistant", "content": "C 程序编译运行成功。Python 程序也运行成功。全部完成。"},
    ]


def main() -> None:
    import os
    client, model = create_client()

    # Use LARGE_COMPACT=1 to test the 48-message failure case
    if os.getenv("REALISTIC_COMPACT"):
        messages = make_realistic_messages_with_long_tool_output()
        print("=== REALISTIC_COMPACT mode: long tool outputs ===\n")
    elif os.getenv("LARGE_COMPACT"):
        messages = make_large_messages_with_tools(6)  # 6*8=48 messages
        print("=== LARGE_COMPACT mode: ~48 messages with tool calls ===\n")
    else:
        messages = make_minimal_messages_with_tools()
        print("=== Minimal mode: 8 messages with tool calls ===\n")

    print("=== Input: formatted for compaction ===\n")
    formatted = _format_messages_for_compaction(messages)
    print(formatted[:1500] + ("..." if len(formatted) > 1500 else ""))
    print(f"\n(Total {len(formatted)} chars, {len(messages)} messages)\n")

    print("=== Calling compact_messages (max_output_tokens=1200) ===\n")
    result = compact_messages(client, model, messages, max_output_tokens=1200)

    # Re-run with debug: capture raw output by calling API directly
    print("=== Raw API call (to see actual response) ===\n")
    try:
        from openai import OpenAI

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": COMPACTOR_SYSTEM_PROMPT},
                {"role": "user", "content": formatted},
            ],
            max_tokens=1200,
            temperature=0.0,
        )
        raw = response.choices[0].message.content or ""
        print("Raw response:")
        print("-" * 40)
        print(raw)
        print("-" * 40)
        print(f"Length: {len(raw)} chars")
        if response.usage:
            print(f"Completion tokens: {response.usage.completion_tokens}")

        parsed = _parse_compacted_output(raw)
        print(f"\nParse result: {len(parsed) if parsed else 0} messages")
        if parsed:
            for i, m in enumerate(parsed):
                print(f"  [{i}] {m['role']}: {m['content'][:80]}...")
        else:
            print("  -> _parse_compacted_output returned None")
            try:
                json.loads(raw.strip())
                print("  -> json.loads succeeded (check role/content filter)")
            except json.JSONDecodeError as e:
                print(f"  -> json.loads failed: {e}")
    except Exception as e:
        print(f"API error: {e}")
        import traceback
        traceback.print_exc()

    print("\n=== compact_messages return value ===")
    print(f"Result: {result}")
    if result:
        print(f"Parsed {len(result)} messages")
    else:
        print("Returned None (compaction failed)")


if __name__ == "__main__":
    main()
