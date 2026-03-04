---
name: core
description: General utility skill for basic math/time tasks and fallback assistant behavior. Use for arithmetic or current-time requests.
tools:
  - calculator
  - get_current_time
triggers:
  - calculate
  - math
  - number
  - time
  - clock
always_on: true
---

# Core Skill

Use utility tools when they clearly help:
- `calculator` for arithmetic/math expressions
- `get_current_time` for date/time queries

Keep answers concise and directly grounded in tool output.
