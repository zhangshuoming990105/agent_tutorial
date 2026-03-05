# Step 07 — CUDA Kernel Development Agent

An autonomous agent that accelerates PyTorch models by implementing custom CUDA C++ extensions. Given a `model.py`, the agent writes CUDA kernels, compiles, verifies correctness, and profiles performance in an iterative loop.

This step is the **non-preemptible baseline** CUDA agent. For queue-based runtime preemption (interrupting autonomous turns with new user input), use Step 08.

## What's New

Built on Step 06 (error recovery, tool use, skill routing), this step adds:

- **CUDA-specific system prompt** with workspace structure and restrictions
- **Task workspace initialisation** — `--task` specifies a directory containing `model.py`; the agent sets up an isolated workspace with the compile/verify/profile infrastructure
- **High-autonomy agent loop** (default 20 steps per turn) for compile → verify → profile cycles
- **CUDA-aware failure classification** — compile errors, correctness failures, and performance gaps each get targeted recovery nudges
- **Shell safety OFF by default** — compile/verify/profile are known-safe within the isolated workspace

## Architecture

```
07_cuda_agent/
├── chatbot.py              # Main CUDA agent (high-autonomy loop)
├── cuda_task.py            # Task workspace lifecycle manager
├── tools.py                # Tool registry (configurable workspace root)
├── context.py              # Token accounting, message management
├── compactor.py            # LLM-based context compaction
├── skill_manager.py        # Skill loading, trigger matching
├── skills/
│   ├── core/SKILL.md       # General utilities (always on)
│   ├── filesystem/SKILL.md # File read/write/search
│   ├── shell/SKILL.md      # Shell execution
│   └── cuda/SKILL.md       # CUDA kernel development guidance (always on)
├── template/               # Fixed workspace infrastructure
│   ├── binding.cpp         # pybind11 module entry point
│   ├── binding_registry.h  # REGISTER_BINDING macro system
│   └── utils/              # compile.py, compile.sh, verification.py, profiling.py
└── task/
    └── example_axpby/      # Sample task: alpha * a + b
        └── model.py
```

## Quick Start

```bash
cd 07_cuda_agent
pip install -r requirements.txt

# Set API key (Ksyun or InfiniAI)
export KSYUN_API_KEY="your-key"

# Run with the example task
python chatbot.py --task task/example_axpby
```

The agent will:
1. Read `model.py` to understand the PyTorch forward pass
2. Write CUDA kernel files in `kernels/` and `model_new.py`
3. Compile with `bash utils/compile.sh`
4. Verify correctness with `python3 -m utils.verification`
5. Profile performance with `python3 -m utils.profiling`
6. Iterate until correct and at least 5% faster than `torch.compile`

## Agent Workflow

```
model.py (input)
    │
    ▼
┌─────────────────┐
│ Read & analyse   │
│ forward pass     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────┐
│ Write kernels/  │────▶│ Write model_new.py│
│ *.cu + *_binding│     │ import cuda_ext   │
└────────┬────────┘     └────────┬─────────┘
         │                       │
         ▼                       ▼
    ┌──────────┐          ┌───────────┐
    │ Compile  │◀─────────│  Fix code │
    └────┬─────┘  fail    └───────────┘
         │ pass
         ▼
    ┌──────────┐          ┌───────────┐
    │ Verify   │◀─────────│ Fix logic │
    └────┬─────┘  fail    └───────────┘
         │ pass
         ▼
    ┌──────────┐          ┌───────────┐
    │ Profile  │◀─────────│ Optimise  │
    └────┬─────┘  slow    └───────────┘
         │ fast
         ▼
      ✅ Done
```

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--task <dir>` | (required) | Task directory containing `model.py` |
| `--workdir <dir>` | `<task>/workdir` | Isolated working directory |
| `--model <name>` | provider default | LLM model to use |
| `--max-agent-steps <n>` | 20 | Max autonomous tool rounds per turn |
| `--safe-shell` | off | Enable shell command approval prompts |
| `--keep-recovery-trace` | off | Keep failed traces in context |
| `--max-tokens <n>` | 128,000 | Context window size |

## Using Your Own Tasks

Create a directory with a `model.py` that defines:
- `Model(nn.Module)` — the PyTorch model to accelerate
- `get_inputs()` — returns sample input tensors
- `get_init_inputs()` — returns constructor arguments

```bash
python chatbot.py --task /path/to/your/task
```

## Next Step

Step 08 builds on this baseline and adds a preemptible input queue plus optional long-shell-command interruption:

- [08_preemptible_cuda_agent](../08_preemptible_cuda_agent/)
