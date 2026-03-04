# Building a General-Purpose LLM Agent from Scratch

A step-by-step tutorial that incrementally builds an autonomous LLM agent — from a simple chatbot to a system resembling Cursor or Claude Code.

Each step is a **self-contained, independently runnable** project. Later steps build upon earlier ones, adding one new core capability at a time.

## Roadmap

| Step | Name | Core Capability |
| ------ | ------ | ---------------- |
| [01](01_basic_chatbot/) | Basic Chatbot | LLM API calls, multi-turn conversation |
| [02](02_context_management/) | Context Management | Token counting, context inspection, debug view |
| [03](03_tool_use/) | Tool Use | Function calling, tool schema, smart compaction, calibrated token tracking |
| [04](04_file_tools/) | File Tools | File read/write/search + dynamic skill-based schema exposure |
| [05](05_shell_tool/) | Shell Tool | Shell command execution with safety approvals |
| [06](06_error_recovery/) | Error Recovery | Task-level retry, autonomous repair, optional failed-trace cleanup |
| 07 | Sub-Agents | Multi-agent architecture, task decomposition, result aggregation |
| 08 | MCP Support | Model Context Protocol, dynamic tool discovery |
| 09 (optional) | Streaming | Streaming output and richer terminal UX |

## Quick Start

```bash
cd 01_basic_chatbot
pip install -r requirements.txt
export INFINI_API_KEY="your-key"
python chatbot.py
```

## Provider

This tutorial uses [InfiniAI](https://cloud.infini-ai.com) as the LLM provider via the OpenAI-compatible API. The code works with any OpenAI-compatible provider — just change `INFINI_BASE_URL`.

## References

- [Building Effective Agents — Anthropic](https://www.anthropic.com/engineering/building-effective-agents)
- [InfiniAI API Docs](https://docs.infini-ai.com/gen-studio/api/)
