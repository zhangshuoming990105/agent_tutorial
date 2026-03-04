# Step 04: File Tools

This step upgrades the agent from "tool-capable" to "codebase-capable".
The model can now inspect and edit files through filesystem tools.

## What's New

- `list_directory` - browse folders
- `read_file` - read text with line numbers
- `write_file` - overwrite/append file content
- `search_files` - search with ripgrep (`rg`)
- `grep_text` - grep-style text search (literal by default)
- `workspace_info` - discover workspace root

Step 03 features are kept:

- tool-call loop
- smart compaction
- calibrated token tracking
- `/history`, `/debug`, `/tokens`
- `/verbose on|off` for detailed token-cost diagnostics
- dynamic skill-based tool exposure (`skills/*/SKILL.md`)

## Setup

```bash
cd 04_file_tools
pip install -r requirements.txt
export INFINI_API_KEY="your-api-key-here"
python chatbot.py
# optional debug mode for autonomy limit:
python chatbot.py --max-agent-steps 5
```

## Example Prompts

- "List files in the project root."
- "Read `03_tool_use/tools.py`."
- "Search for `def execute_tool` under `03_tool_use`."
- "Create `notes.txt` with a short TODO list."
- "/verbose on then /tokens"
- "/skills"
- "/skill filesystem on"

## Design Notes

### Tool Exposure to the LLM

This step uses **dynamic skill routing**.
Tools are grouped in `skills/*/SKILL.md`, and only tools from active skills are
exposed via `tools=` on each API call.

The model still decides:

- whether to call a tool
- which tool to call
- what arguments to use

### Workspace Safety Boundary

File tools only allow paths inside workspace root (`agent_tutorial/`).
Any path outside this boundary is rejected.

### Context vs Billing Tokens

`/tokens` now separates:

- **Managed**: message tokens in managed conversation state
- **Overhead**: active tool schema + skill prompt tokens for current request
- **Effective**: API-calibrated estimate used for context budgeting

This keeps context management and billing analysis explicit rather than conflated.

### Why `search_files` Uses `rg`

`rg` is fast, recursive, and practical for code search at scale. This is the
first time the agent can perform codebase-wide retrieval before deciding edits.

## Files

- `chatbot.py` - REPL + tool loop + compaction flow
- `tools.py` - tool registry + filesystem tools + demo tools
- `context.py` - context tracking + debug formatting
- `compactor.py` - structured context compaction

## What's Next

In [Step 05](../05_shell_tool/), we will add shell execution (`run_command`) with timeout and basic safety checks.
