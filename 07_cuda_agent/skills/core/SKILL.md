---
name: core
description: General utility skill for basic math/time tasks and fallback assistant behavior. Use for arithmetic or current-time requests.
tools:
  - calculator
  - get_current_time
  - shell_policy_status
triggers:
  - calculate
  - math
  - number
  - time
  - clock
  - policy
always_on: true
---

# Core Skill

Use utility tools when they clearly help:
- `calculator` for arithmetic/math expressions
- `get_current_time` for date/time queries
- `shell_policy_status` when user asks about shell permissions

Keep answers concise and grounded in tool output.
