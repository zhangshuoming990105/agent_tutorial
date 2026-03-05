#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_DIR="${ROOT_DIR}/.live_session"
FIFO_PATH="${STATE_DIR}/stdin.fifo"
LOG_PATH="${STATE_DIR}/stdout.log"
PID_PATH="${STATE_DIR}/pid"
KEEPER_PID_PATH="${STATE_DIR}/fifo_keeper.pid"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/live_session.sh start [chatbot args...]
  bash scripts/live_session.sh send "<message>"
  bash scripts/live_session.sh status
  bash scripts/live_session.sh stop
  bash scripts/live_session.sh paths

Examples:
  bash scripts/live_session.sh start --task level1/003 --preempt-shell-kill
  bash scripts/live_session.sh send "what's current status?"
  bash scripts/live_session.sh send "quit"
EOF
}

ensure_state_dir() {
  mkdir -p "${STATE_DIR}"
}

is_running() {
  if [[ ! -f "${PID_PATH}" ]]; then
    return 1
  fi
  local pid
  pid="$(<"${PID_PATH}")"
  [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null
}

start_fifo_keeper() {
  (
    exec 9<>"${FIFO_PATH}"
    while :; do
      sleep 3600
    done
  ) &
  echo "$!" > "${KEEPER_PID_PATH}"
}

stop_fifo_keeper() {
  if [[ -f "${KEEPER_PID_PATH}" ]]; then
    local kpid
    kpid="$(<"${KEEPER_PID_PATH}")"
    if [[ -n "${kpid}" ]]; then
      kill "${kpid}" 2>/dev/null || true
    fi
    rm -f "${KEEPER_PID_PATH}"
  fi
}

require_running() {
  if ! is_running; then
    echo "No live session running." >&2
    echo "Start one with: bash scripts/live_session.sh start ..." >&2
    exit 1
  fi
}

cmd_start() {
  ensure_state_dir
  if is_running; then
    echo "A live session is already running (pid $(<"${PID_PATH}"))." >&2
    exit 1
  fi

  rm -f "${FIFO_PATH}" "${LOG_PATH}" "${PID_PATH}"
  mkfifo "${FIFO_PATH}"
  start_fifo_keeper

  (
    cd "${ROOT_DIR}"
    PYTHONUNBUFFERED=1 stdbuf -oL -eL python -u chatbot.py "$@" < "${FIFO_PATH}" 2>&1 | tee -a "${LOG_PATH}"
  ) &
  local pid=$!
  echo "${pid}" > "${PID_PATH}"

  echo "started_pid=${pid}"
  echo "fifo=${FIFO_PATH}"
  echo "log=${LOG_PATH}"
}

cmd_send() {
  require_running
  local msg="${*:-}"
  if [[ -z "${msg}" ]]; then
    echo "send requires a non-empty message." >&2
    exit 1
  fi
  printf '%s\n' "${msg}" > "${FIFO_PATH}"
  echo "sent: ${msg}"
}

cmd_status() {
  if is_running; then
    echo "running=true"
    echo "pid=$(<"${PID_PATH}")"
  else
    echo "running=false"
    stop_fifo_keeper
  fi
  echo "fifo=${FIFO_PATH}"
  echo "log=${LOG_PATH}"
}

cmd_stop() {
  if is_running; then
    printf 'quit\n' > "${FIFO_PATH}" || true
    local pid
    pid="$(<"${PID_PATH}")"
    sleep 0.2
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
    rm -f "${PID_PATH}"
    stop_fifo_keeper
    echo "stopped"
  else
    echo "no running session"
    stop_fifo_keeper
  fi
}

cmd_paths() {
  echo "fifo=${FIFO_PATH}"
  echo "log=${LOG_PATH}"
  echo "pid_file=${PID_PATH}"
  echo "keeper_pid_file=${KEEPER_PID_PATH}"
}

main() {
  local sub="${1:-}"
  case "${sub}" in
    start)
      shift
      cmd_start "$@"
      ;;
    send)
      shift
      cmd_send "$@"
      ;;
    status)
      cmd_status
      ;;
    stop)
      cmd_stop
      ;;
    paths)
      cmd_paths
      ;;
    *)
      usage
      exit 1
      ;;
  esac
}

main "$@"
