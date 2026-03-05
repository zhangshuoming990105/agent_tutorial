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
- `README.md`：项目整体说明与 Live E2E Test Workflow 概述
