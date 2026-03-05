# Agent Live E2E Test

本文档定义 **live e2e test** 的标准流程，供人类和 AI agent 共同使用。  
当你说「跑 live e2e test」「按 agent_test 测试」或类似指令时，应遵循本文档。

## 目的

- **统一入口**：人类和 AI 都用同一套流程，避免歧义
- **真实环境**：单进程、FIFO 注入、完整日志，复现真实运行行为
- **可观测**：所有输出写入 `logs/<timestamp>.log`，便于事后排查

## 前置条件

```bash
cd 08_preemptible_cuda_agent
pip install -r requirements.txt
export KSYUN_API_KEY="your-key"   # 或对应环境的 API key
```

## 标准流程

### 1. 启动会话

```bash
bash scripts/live_session.sh stop || true
bash scripts/live_session.sh start
```

可选参数示例：`bash scripts/live_session.sh start --task level1/001 --preempt-shell-kill`

### 2. 发送输入

```bash
bash scripts/live_session.sh send "<消息内容>"
```

### 3. 判断 Agent 是否完成

**关键信号**：当 log 中出现 `>>> Ready for input.` 时，表示当前 turn 已结束，可以发送下一条消息。

完整的一轮通常包含：
- `You: <用户输入>`
- （可选）`[skills] active: ...`
- （可选）`-> Calling tool: ...` / `<- Result: ...`
- `Assistant: <回复>`
- `[tokens: prompt=..., completion=..., total=...]`
- `>>> Ready for input.`

**轮询策略**：若未看到 `>>> Ready for input.`，建议每隔 15 秒查看一次 log，直到出现该行再发下一条。

### 4. 查看日志

- 实时流：`.live_session/stdout.log`
- 持久化：`logs/<timestamp>.log`（会话结束时写入）

```bash
tail -f .live_session/stdout.log
# 或
tail -50 logs/$(ls -t logs/*.log | head -1 | xargs basename)
```

### 5. 结束会话

```bash
bash scripts/live_session.sh send "quit"
```

## 常用测试场景

### 场景 A：基础 slash 命令

验证 `/help`、`/task`、`/preempt`、`/tokens`、`/skills` 等命令是否正常、输出是否写入 log。

```bash
bash scripts/live_session.sh send "/help"
# 等待 >>> Ready for input.
bash scripts/live_session.sh send "/task"
# 等待 >>> Ready for input.
bash scripts/live_session.sh send "quit"
```

### 场景 B：任务加载与工具调用

验证 `/task load`、任务加载后的工具调用（如 `get_current_time`、`run_shell` 等）。

```bash
bash scripts/live_session.sh send "/task load 01"
# 等待 >>> Ready for input.
bash scripts/live_session.sh send "what time is it?"
# 等待 >>> Ready for input.
bash scripts/live_session.sh send "quit"
```

### 场景 C：完整 CUDA 任务

验证从加载任务到编译、验证、profiling 的完整流程。

```bash
bash scripts/live_session.sh send "/task load 01"
# 等待 >>> Ready for input.
bash scripts/live_session.sh send "start"
# 等待较长时间（编译+验证+profiling 可能 1–2 分钟），直到出现 >>> Ready for input.
bash scripts/live_session.sh send "quit"
```

### 场景 D：连续工具调用（非抢占）

验证 agent 在一轮内连续执行多种工具调用的能力。**非抢占模式**：每次发送后必须等待 `>>> Ready for input.` 再发下一条。

**启动**（不加载 task，chat-first 模式）：

```bash
bash scripts/live_session.sh stop || true
bash scripts/live_session.sh start
```

**步骤 1–7**：发送一条综合指令，让 agent 在一轮内完成以下操作（连续工具调用）。

> ⚠️ **注意**：`live_session.sh send` 按行读取，多行消息会被拆成多条输入。请使用**单行**指令，例如：

```
请按顺序完成：1) 在 temp 下创建以时间戳命名的文件夹（如 temp/20260306_123456）作为工作目录；2) 在里面写几个 example 的 C 和 Python 程序；3) 连续读这些文件确认内容；4) 对其中一个文件用 append 模式追加内容；5) 用 gcc 编译 C；6) 运行 C 的 binary；7) 运行 Python 程序。完成后简要总结。
```

等待 `>>> Ready for input.`。

**步骤 8**：测试 slash 命令

```bash
bash scripts/live_session.sh send "/help"
# 等待 >>> Ready for input.
bash scripts/live_session.sh send "/debug"
# 等待 >>> Ready for input.
bash scripts/live_session.sh send "/tokens"
# 等待 >>> Ready for input.
```

**步骤 9**：压缩上下文

```bash
bash scripts/live_session.sh send "/compact"
# 等待 >>> Ready for input.
```

**步骤 10**：再次查看 debug，验证上下文是否被压缩

```bash
bash scripts/live_session.sh send "/debug"
# 等待 >>> Ready for input.
bash scripts/live_session.sh send "quit"
```

**验证点**：
- [ ] 步骤 1–7：`write_file`、`read_file`、`write_file mode=append`、`run_shell`（gcc、运行 binary、python）均有调用
- [ ] `/help`、`/debug`、`/tokens` 输出正确
- [ ] `/compact` 后 `/debug`：若压缩成功，上下文消息数或 token 数应减少；若显示 "Compaction failed"，则上下文不变（可能因当前 token 较少或 API 限制）

### 场景 E：20 轮多工具连续对话 + /compact high（自动化）

验证多轮工具调用积累后的上下文压缩效果，同时作为完整的端到端回归测试。

**方式 1 — 一键运行（推荐）**：

`scripts/run_test.sh` 会自动启动 agent（使用 `gpt-oss-120b`）、调用 `run_20_turns.sh`、执行断言，最后退出并打印 PASS/FAIL：

```bash
cd 08_preemptible_cuda_agent
bash scripts/run_test.sh               # 默认 model: gpt-oss-120b
bash scripts/run_test.sh --model mco-4 # 指定其他 model
```

**方式 2 — 分步手动**：

```bash
# 启动 agent（模型显式指定）
bash scripts/live_session.sh stop || true
bash scripts/live_session.sh start --model gpt-oss-120b --compact-model gpt-oss-120b

# 等待 >>> Ready for input. 后运行 20 轮脚本
bash scripts/run_20_turns.sh
# run_20_turns.sh 结束后自动发送 /tokens、/compact high、/debug raw

bash scripts/live_session.sh send "quit"
```

**`run_20_turns.sh` 执行的 20 轮内容**：

| 轮次 | 操作 |
|:---:|------|
| 1 | `get_current_time` 工具调用 |
| 2–4 | 写入 `hello.c`、`factorial.c`，读取确认 |
| 5–6 | gcc 编译两个 C 程序 |
| 7–8 | 运行两个 C 二进制 |
| 9–10 | 写入并运行 `fib.py` |
| 11–13 | append 写 `hello.c`，确认，重新编译运行 |
| 14–15 | 写入并运行 `primes.py` |
| 16 | `list_directory` 列出目录 |
| 17 | `calculator` 工具计算 |
| 18–20 | 写入并运行 `stats.py` |
| 结束 | `/tokens` → `/compact high` → `/debug raw` |

**断言**（`run_test.sh` 自动验证）：
- [ ] Ready 信号数 ≥ 23（20 turns + /tokens + /compact + /debug）
- [ ] `write_file`、`read_file`、`run_shell` 均被调用
- [ ] `/compact high` 成功：log 中出现 `Compressed N old messages`
- [ ] 无 compact parse error
- [ ] `/debug raw` 输出包含 `"role":` JSON 字段

## 验证清单

- [ ] slash 命令输出写入 `logs/<timestamp>.log`
- [ ] 工具调用 `-> Calling tool` / `<- Result` 写入 log
- [ ] 每轮结束后出现 `>>> Ready for input.`
- [ ] 多轮对话顺序正确，无乱序或丢失

## 对 AI Agent 的说明

若你被要求「进行 live e2e test」或「按 agent_test 测试」：

1. **按本文档执行**：启动 → 发送 → 等待 `>>> Ready for input.` → 再发 → 结束
2. **不要急于发下一条**：必须等到出现 `>>> Ready for input.` 再发，否则会进入队列、可能触发 preempt
3. **验证 log 内容**：在 `quit` 后检查 `logs/` 下最新 log，确认 slash 命令和工具调用均有记录
4. **若测试目标不明确**：可默认执行「场景 A + 场景 B」作为最小验证集

## 相关文件

- `scripts/live_session.sh`：会话控制脚本
- `scripts/run_20_turns.sh`：20 轮对话脚本（需已有运行中的 session）
- `scripts/run_test.sh`：完整端到端测试（自动起 agent + 20 轮 + 断言）
- `README.md`：项目整体说明与 Live E2E Test Workflow 概述
