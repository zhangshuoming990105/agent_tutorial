---
name: shell
description: Command-line execution skill with safety approvals. Use for terminal commands (git, npm, python, ls, pwd, etc.).
tools:
  - run_shell
  - shell_policy_status
triggers:
  - shell
  - command
  - terminal
  - bash
  - zsh
  - git
  - npm
  - pip
  - python
  - run
  - execute
always_on: false
---

# Shell Skill

Use `run_shell` for command execution.

Safety behavior:

- Safe shell mode is ON by default.
- If safe shell mode is ON, `run_shell` will ask user confirmation for new commands:
  1) allow once
  2) always allow (adds to allowlist)
  3) deny (adds to denylist)
- If safe shell mode is OFF, run needed commands directly without extra assistant-level confirmation.

When user asks about shell permissions, call `shell_policy_status`.
