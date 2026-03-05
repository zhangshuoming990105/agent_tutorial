# Step 08 — Preemptible CUDA Agent

Step 08 extends the CUDA workflow from Step 07 with **queue-based user preemption**:

- while the agent runs autonomous compile/verify/profile loops, new user input can interrupt the loop at safe boundaries
- optional mixed mode can also terminate long-running shell commands on preempt

## What's New vs Step 07

- **Background input queue** with preemption-aware chat loop
- **Soft preemption by default** (switch turns at safe points between autonomous steps)
- **Optional shell kill on preempt** for long `run_shell` commands
- **Runtime task-context injection** retained (`/task load`, `/task reload`, `/task inject`)
- **Global task template fallback** retained (`task/TASK_TEMPLATE.md`)

## Architecture

```
08_preemptible_cuda_agent/
├── chatbot.py              # Preemptible autonomous loop + input queue
├── runtime_state.py        # Shared preempt/interruption flags
├── cuda_task.py            # Task workspace lifecycle manager
├── tools.py                # Tool registry + interrupt-aware run_shell
├── context.py              # Token accounting, message management
├── compactor.py            # LLM-based context compaction
├── skill_manager.py        # Skill loading, trigger matching
├── skills/
├── template/
└── task/
```

## Quick Start

```bash
cd 08_preemptible_cuda_agent
pip install -r requirements.txt
export KSYUN_API_KEY="your-key"

# Start directly on a task
python chatbot.py --task task/example_axpby

# Or chat-first mode and load task later
python chatbot.py
```

## Live E2E Test Workflow (Default)

For real agent debugging in Step 08, use a **single long-lived chatbot process** and drive it by appending inputs through `live_session.sh`.
All assistant-initiated real tests should follow this workflow by default.

This is now the preferred test method for end-to-end validation because it reproduces real runtime behavior:

- one persistent chatbot process
- async input injection via FIFO
- fully observable `You:` / `Assistant:` turn logs
- preemption and non-preemption behavior both testable in one setup

### Standard Session Commands

```bash
cd 08_preemptible_cuda_agent

# start a clean chat-first session
bash scripts/live_session.sh stop || true
bash scripts/live_session.sh start

# send one user turn
bash scripts/live_session.sh send "who are you?"

# inspect live paths/status
bash scripts/live_session.sh status
bash scripts/live_session.sh paths

# end session
bash scripts/live_session.sh send "quit"
```

### Non-preemptive Multi-round Validation (Recommended)

Use this when validating baseline conversation behavior before testing interrupt paths:

1. send exactly one user message (`send "..."`)
2. poll logs in fixed intervals (recommended: every 15 seconds)
3. wait until the current round is complete (`Assistant:` reply + `[tokens: ...]`)
4. send the next user message
5. if the reply is off-target, send a correction like a human would, then continue

This verifies:

- queue input is consumed correctly in-order
- per-turn completion semantics are stable without forced preemption
- tool calls (date/math/system checks) remain observable and debuggable in real time

### Live Log Visibility

`scripts/live_session.sh` runs Python in unbuffered mode, so output is flushed continuously to:

- `.live_session/stdout.log` (live session stream)
- `logs/<timestamp>.log` (agent run log)

You can keep either file open during testing to watch each turn appear in real time.

## Preemption Modes

- **Default soft preemption**:
  - new user input queues immediately
  - autonomous loop is interrupted at the next safe boundary
  - queued user input becomes the next turn context
- **Optional mixed mode (shell kill)**:
  - enable with `--preempt-shell-kill`
  - if preempt arrives during a long shell command, command is terminated

In-session controls:

- `/preempt` show current settings
- `/preempt shell-kill on|off` toggle shell termination on preempt

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--task <dir>` | (optional) | Task directory containing `model.py`; can also be loaded later with `/task load` |
| `--workdir <dir>` | `<task>/workdir` | Isolated working directory |
| `--model <name>` | provider default | LLM model to use |
| `--max-agent-steps <n>` | 30 | Max autonomous tool rounds per turn |
| `--safe-shell` | off | Enable shell command approval prompts |
| `--keep-recovery-trace` | off | Keep failed traces in context |
| `--preempt-shell-kill` | off | Kill running shell commands when preempt input arrives |
| `--max-tokens <n>` | 128,000 | Context window size |

## Task Context Workflow

Case task docs:

- preferred: case-level `task.md` or `TASK.md`
- fallback: `task/TASK_TEMPLATE.md`

Useful commands:

- `/task load <specifier>` initialize/switch task workspace
- `/task reload` reload latest history + task context
- `/task inject` inject task context into ongoing conversation

## Relation to Step 07

- Step 07 remains the stable non-preemptible baseline CUDA agent.
- Step 08 is the preemptible variant for interactive takeover during autonomous runs.
