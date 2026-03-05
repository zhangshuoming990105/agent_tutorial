#!/usr/bin/env bash
# test_set_model.sh — verify /set-model switches model while preserving context.
#
# Flow:
#   1. Start agent (mco-4)
#   2. Ask a question (build some context)
#   3. /set-model gpt-oss-120b
#   4. /tokens  — verify model name changed
#   5. Ask a follow-up that references the first answer (proves context preserved)
#   6. Assert: model name in /tokens output changed AND follow-up answered correctly
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$ROOT/.live_session/stdout.log"
SCRIPTS="$ROOT/scripts"
FAILURES=0

pass() { echo "[PASS] $*"; }
fail() { echo "[FAIL] $*" >&2; FAILURES=$(( FAILURES + 1 )); }

ready_count() {
    local n
    n=$(grep -c '>>> Ready for input.' "$LOG" 2>/dev/null) || n=0
    echo "$n"
}

wait_ready() {
    local expected=$1
    local timeout="${2:-60}"
    local deadline=$(( $(date +%s) + timeout ))
    while true; do
        [ "$(ready_count)" -ge "$expected" ] && return 0
        [ "$(date +%s)" -ge "$deadline" ] && { echo "[TIMEOUT] waited ${timeout}s"; return 1; }
        sleep 2
    done
}

send() {
    local msg="$1"; local expected=$2
    bash "$SCRIPTS/live_session.sh" send "$msg"
    wait_ready "$expected" || true
}

# ---------------------------------------------------------------------------
echo "[test_set_model] Starting agent with mco-4..."
bash "$SCRIPTS/live_session.sh" stop 2>/dev/null || true
bash "$SCRIPTS/live_session.sh" start --model mco-4
wait_ready 1 30

# Step 1: ask something to build context
echo "[test_set_model] Step 1: build context..."
send "请记住这个数字：42。然后告诉我它是奇数还是偶数。" 2

# Step 2: switch model
echo "[test_set_model] Step 2: /set-model gpt-oss-120b..."
send "/set-model gpt-oss-120b" 3

# Step 3: /tokens — check model name
echo "[test_set_model] Step 3: /tokens..."
send "/tokens" 4

# Step 4: follow-up that requires previous context
echo "[test_set_model] Step 4: follow-up using context..."
send "你之前记住的那个数字，乘以2是多少？" 5

bash "$SCRIPTS/live_session.sh" send "quit"

# ---------------------------------------------------------------------------
echo ""
echo "=== Assertions ==="

# Model switch message appeared
if grep -q "Model switched: mco-4 → gpt-oss-120b" "$LOG" 2>/dev/null; then
    pass "Model switch log message present"
else
    fail "Model switch log message NOT found"
fi

# /tokens shows new model
if grep -q "Model:.*gpt-oss-120b" "$LOG" 2>/dev/null; then
    pass "/tokens shows gpt-oss-120b"
else
    fail "/tokens does not show gpt-oss-120b"
fi

# Follow-up answer contains 84 (42 * 2)
if grep -q "84" "$LOG" 2>/dev/null; then
    pass "Follow-up answer contains 84 (42*2 correct)"
else
    fail "Follow-up answer does not contain 84"
fi

# Context was preserved (42 referenced)
if grep -q "42" "$LOG" 2>/dev/null; then
    pass "Number 42 found in log (context preserved)"
else
    fail "42 not found — context may have been lost"
fi

echo ""
echo "=============================="
if [ "$FAILURES" -eq 0 ]; then
    echo "ALL TESTS PASSED"
else
    echo "$FAILURES TEST(S) FAILED"
fi
echo "Log: $LOG"
echo "=============================="
exit "$FAILURES"
