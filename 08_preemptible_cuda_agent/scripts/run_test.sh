#!/usr/bin/env bash
# run_test.sh — Full end-to-end test: start agent (gpt-oss-120b) + run 20 turns + /compact high.
#
# Usage:
#   cd 08_preemptible_cuda_agent
#   bash scripts/run_test.sh              # default model: gpt-oss-120b
#   bash scripts/run_test.sh --model mco-4
#
# Exit codes:
#   0  all assertions passed
#   1  one or more assertions failed
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$ROOT/.live_session/stdout.log"
SCRIPTS="$ROOT/scripts"

MODEL="gpt-oss-120b"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

info()  { echo "[run_test] $*"; }
pass()  { echo "[PASS] $*"; }
fail()  { echo "[FAIL] $*" >&2; FAILURES=$((FAILURES + 1)); }

WAIT_TIMEOUT="${WAIT_TIMEOUT:-120}"  # seconds per wait_ready call

wait_ready() {
    local expected=$1
    local timeout="${2:-$WAIT_TIMEOUT}"
    local deadline=$(( $(date +%s) + timeout ))
    local count
    while true; do
        count=$(grep -c '>>> Ready for input.' "$LOG" 2>/dev/null) || count=0
        [ "$count" -ge "$expected" ] && return 0
        if [ "$(date +%s)" -ge "$deadline" ]; then
            info "wait_ready: timeout after ${timeout}s waiting for count>=$expected (got $count)"
            return 1
        fi
        sleep 2
    done
}

send() {
    local msg="$1"
    local expected=$2
    bash "$SCRIPTS/live_session.sh" send "$msg"
    wait_ready "$expected" || true
}

FAILURES=0

# ---------------------------------------------------------------------------
# 1. start agent
# ---------------------------------------------------------------------------
info "Stopping any existing session..."
bash "$SCRIPTS/live_session.sh" stop 2>/dev/null || true

info "Starting agent (model=$MODEL)..."
bash "$SCRIPTS/live_session.sh" start --model "$MODEL" --compact-model "$MODEL"
wait_ready 1
info "Agent ready."

# ---------------------------------------------------------------------------
# 2. run 20 turns (delegates to run_20_turns.sh which requires session running)
# ---------------------------------------------------------------------------
info "Running 20-turn conversation..."
bash "$SCRIPTS/run_20_turns.sh"

# ---------------------------------------------------------------------------
# 3. wait for all 23 ready signals then run assertions
# ---------------------------------------------------------------------------
info "Waiting for all 23 ready signals (20 turns + /tokens + /compact + /debug raw)..."
wait_ready 23 300  # /compact high may take up to ~60s extra
info "Running assertions..."

# All 23 ready signals present
READY_COUNT=$(grep -c '>>> Ready for input.' "$LOG" 2>/dev/null) || READY_COUNT=0
if [ "$READY_COUNT" -ge 23 ]; then
    pass "Ready signals: $READY_COUNT >= 23"
else
    fail "Ready signals: $READY_COUNT < 23"
fi

# Tool calls were made
for tool in write_file read_file run_shell; do
    if grep -q "Calling tool: $tool" "$LOG" 2>/dev/null; then
        pass "Tool called: $tool"
    else
        fail "Tool NOT called: $tool"
    fi
done

# Compact was attempted
if grep -q "Compacting .* old messages" "$LOG" 2>/dev/null; then
    pass "Compact was attempted: $(grep 'Compacting .* old messages' "$LOG" | tail -1)"
else
    fail "Compact was never attempted"
fi

# Compact succeeded or failed gracefully (rate-limit is acceptable)
if grep -q "Compressed .* old messages" "$LOG" 2>/dev/null; then
    COMPACT_LINE=$(grep "Compressed .* old messages" "$LOG" | tail -1)
    pass "Compact succeeded: $COMPACT_LINE"
elif grep -q "rate.limit\|429\|retrying" "$LOG" 2>/dev/null; then
    pass "Compact skipped due to rate limit (acceptable)"
else
    fail "Compact failed for unexpected reason"
fi

# No parse errors
if grep -q "Compaction parse failed" "$LOG" 2>/dev/null; then
    fail "Compact parse error found in log"
else
    pass "No compact parse errors"
fi

# debug raw output present
if grep -q '"role":' "$LOG" 2>/dev/null; then
    pass "/debug raw produced JSON output"
else
    fail "/debug raw produced no JSON output"
fi

# ---------------------------------------------------------------------------
# 4. quit + summary
# ---------------------------------------------------------------------------
bash "$SCRIPTS/live_session.sh" send "quit"

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
