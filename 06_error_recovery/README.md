# Step 06: Error Recovery (Task-Level Autonomous Repair)

This step adds an explicit recovery layer to the agent loop: when tools fail, the agent keeps working autonomously (within `max-agent-steps`) to diagnose, retry, and repair.

## What's New

- Task-level `RecoveryState` tracks failure history for the current user task
- Tool failures detected structurally (generic `Error/Denied`, non-zero `run_shell` exit codes)
- Recovery nudge injected only when needed, guiding the model to fix root causes
- Guardrail against premature stop:
  - if task looks action-oriented but no tool action happened yet, keep going
  - if unresolved failure exists, keep going
  - if model tries to hand off to user prematurely, suppress and continue
- Optional cleanup: after task success, failed intermediate tool traces removed from context
- `run_shell` now accepts a `cwd` parameter (defaults to workspace root), eliminating `cd` failures
- System prompt includes workspace root path so the LLM knows where it is
- `test_run.py` non-interactive test harness with real-time colored output

## Setup

```bash
cd 06_error_recovery
pip install -r requirements.txt
export INFINI_API_KEY="your-api-key-here"
python chatbot.py
```

Optional:

```bash
python chatbot.py --max-agent-steps 8
python chatbot.py --unsafe-shell
python chatbot.py --keep-recovery-trace   # do not prune failed intermediates
```

## Test Harness

Run a single-turn task without interactive input:

```bash
python test_run.py                         # default: C-to-Rust translation task
python test_run.py "your custom prompt"    # any task
```

Output is real-time with colored tool call traces, failure/success status, and timing.

## Recovery Architecture

For each user task/turn:

1. Agent enters autonomous loop (bounded by `max-agent-steps`)
2. Execute tool calls
3. If any tool fails:
   - classify as unresolved failure
   - feed compact recovery guidance into next round
   - continue autonomously
4. If final response is still procedural (plan/confirm) or failure is unresolved, suppress and continue
5. On success:
   - return final answer
   - optionally prune failed intermediate traces from context (`/recovery on`)
6. If step budget is exhausted:
   - surface a clear stuck message plus latest failure summary

## Slash Commands

- `/recovery on|off` - toggle failed-trace cleanup after successful task completion
- `/workspace` - show current per-turn workspace hint
- `/shell-safe on|off` - toggle safe shell mode
- `/shell-policy` - print current allowlist/denylist
- plus all previous commands:
  - `/tokens`, `/verbose`, `/history`, `/debug`, `/compact`
  - `/skills`, `/skill <name> on|off`

## Files

- `chatbot.py` - recovery loop, retry nudges, autonomy guardrails, success cleanup
- `context.py` - context/token accounting + failed trace pruning helper
- `tools.py` - tools with `run_shell(cwd=...)` and shell policy persistence
- `skill_manager.py` - dynamic skill loading/routing
- `skills/` - `core`, `filesystem`, `shell`
- `test_run.py` - non-interactive test harness
- `task/code_translation/example.c` - sample task for testing

## What's Next

In Step 07, we'll implement sub-agents for decomposing complex tasks and aggregating results.
