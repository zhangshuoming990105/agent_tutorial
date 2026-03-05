#!/usr/bin/env bash
# Send 20 turns sequentially, waiting for >>> Ready for input. between each.
# Each turn has a per-turn timeout (default 90s); on timeout the turn is
# marked SKIP and the script continues so the overall run never hangs.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$ROOT/.live_session/stdout.log"
TURN_TIMEOUT="${TURN_TIMEOUT:-90}"   # seconds to wait per turn before skipping

SKIPPED=0

ready_count() {
    local n
    n=$(grep -c '>>> Ready for input.' "$LOG" 2>/dev/null) || n=0
    echo "$n"
}

wait_ready() {
    # Wait until the ready count reaches $1, or until TURN_TIMEOUT seconds pass.
    local expected=$1
    local deadline=$(( $(date +%s) + TURN_TIMEOUT ))
    while true; do
        [ "$(ready_count)" -ge "$expected" ] && return 0
        if [ "$(date +%s)" -ge "$deadline" ]; then
            echo "    [TIMEOUT] turn did not complete within ${TURN_TIMEOUT}s — skipping"
            SKIPPED=$(( SKIPPED + 1 ))
            return 1
        fi
        sleep 2
    done
}

send() {
    local msg="$1"
    local expected=$2
    echo ">>> Sending turn $expected: ${msg:0:60}..."
    bash "$ROOT/scripts/live_session.sh" send "$msg"
    wait_ready "$expected" || true   # continue even on timeout
    echo "    done (ready=$(ready_count))"
}

# Start at ready count 1 (initial)
send "现在几点？用工具查一下" 2
send "在 temp/t20 下写一个 hello.c，打印 Hello World" 3
send "在 temp/t20 下写一个 factorial.c，计算 0 到 10 的阶乘" 4
send "读 temp/t20/hello.c 和 temp/t20/factorial.c 确认内容" 5
send "用 gcc 编译 temp/t20/hello.c 生成 temp/t20/hello" 6
send "用 gcc 编译 temp/t20/factorial.c 生成 temp/t20/factorial" 7
send "运行 temp/t20/hello" 8
send "运行 temp/t20/factorial" 9
send "在 temp/t20 下写一个 fib.py，计算斐波那契数列前 20 项并打印" 10
send "运行 temp/t20/fib.py" 11
send "对 temp/t20/hello.c 用 append 模式追加一行注释 /* compiled and tested */" 12
send "读 temp/t20/hello.c 确认追加成功" 13
send "重新编译 temp/t20/hello.c，再运行一次" 14
send "在 temp/t20 下写一个 primes.py，输出 100 以内的质数" 15
send "运行 temp/t20/primes.py" 16
send "列出 temp/t20 目录下所有文件" 17
send "计算 sqrt(2) + pi 的值，用 calculator 工具" 18
send "在 temp/t20 下写一个 stats.py，生成 20 个随机数并计算均值和标准差" 19
send "运行 temp/t20/stats.py" 20

echo ""
echo "=== 20 turns done. Sending /tokens ==="
bash "$ROOT/scripts/live_session.sh" send "/tokens"
wait_ready 21

echo "=== Sending /compact high ==="
bash "$ROOT/scripts/live_session.sh" send "/compact high"
wait_ready 22

echo "=== Sending /debug raw ==="
bash "$ROOT/scripts/live_session.sh" send "/debug raw"
wait_ready 23

echo "=== Done (skipped=$SKIPPED) ==="
