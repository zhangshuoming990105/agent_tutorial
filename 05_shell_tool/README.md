# Step 05: Shell Tool (Safety-First)

This step adds shell command execution to the agent with Cursor-like safety behavior.

## What's New

- `run_shell` tool for terminal command execution
- **Safe shell mode flag** (`--unsafe-shell` to disable, default is safe)
- When safe mode is ON, new shell commands require user confirmation:
  1. allow once
  2. always allow (add to allowlist)
  3. deny (add to denylist)
- Persistent allowlist/denylist (`05_shell_tool/.shell_policy.json`)
- `shell_policy_status` tool and slash commands for shell policy inspection
- `max-agent-steps` mechanism (default 5), with proactive human-help fallback
- In `--unsafe-shell` mode, the loop favors autonomous tool execution (no procedural "yes/no" handoff) until completion or max-step limit

## Setup

```bash
cd 05_shell_tool
pip install -r requirements.txt
export INFINI_API_KEY="your-api-key-here"
python chatbot.py
```

Optional:

```bash
python chatbot.py --max-agent-steps 5
python chatbot.py --unsafe-shell   # disable safety confirmations (not recommended)
```

## Shell Safety Model

When safe mode is enabled:

- Any new shell command is blocked until user confirms
- User can choose:
  - **allow once**: execute now only
  - **always allow**: execute and add exact command to allowlist
  - **deny**: block and add exact command to denylist
- Future requests for allowlisted commands run without prompt
- Future requests for denylisted commands are blocked immediately

## Slash Commands

- `/shell-safe on|off` - toggle safe shell mode
- `/shell-policy` - print current allowlist/denylist
- plus all previous commands:
  - `/tokens`, `/verbose`, `/history`, `/debug`, `/compact`
  - `/skills`, `/skill <name> on|off`

## Files

- `chatbot.py` - main loop, skills routing, max-step control
- `tools.py` - all tools including `run_shell` and shell policy persistence
- `context.py` - context/token accounting and debug views
- `skill_manager.py` - dynamic skill loading/routing
- `skills/` - `core`, `filesystem`, `shell`

## What's Next

In [Step 06](../06_error_recovery/), we'll add task-level error recovery so the agent can diagnose failures, retry autonomously, and optionally clean failed traces after success.
