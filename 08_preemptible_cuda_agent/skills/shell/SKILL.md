---
name: shell
description: Command-line execution skill for CUDA compilation, verification, and profiling. Shell safety is OFF by default for this agent.
tools:
  - run_shell
  - shell_policy_status
triggers:
  - shell
  - command
  - terminal
  - bash
  - compile
  - run
  - execute
always_on: false
---

# Shell Skill

Use `run_shell` for command execution. Common CUDA workflow commands:

```bash
bash utils/compile.sh              # Compile CUDA extension
python3 -m utils.verification      # Correctness check (5 rounds)
python3 -m utils.profiling         # Performance benchmark
```

Shell safety is OFF by default for the CUDA agent since compile/verify/profile are known-safe operations within the isolated task workspace.

When shell safety is ON, `run_shell` will ask user confirmation for new commands:
  1) allow once
  2) always allow (adds to allowlist)
  3) deny (adds to denylist)
