#!/usr/bin/env bash
set -euo pipefail

# SPS watchdog for long overnight runs.
# Restarts CARLA + trainer when SPS falls below a threshold.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CHECKPOINT="${1:-${REPO_ROOT}/outputs/checkpoints/tfv6_resnet34}"
LOGDIR="${2:-${REPO_ROOT}/outputs/rl_logs}"
EXP_NAME="${3:-TFV6_PPO_SMOKE_OVERNIGHT}"

RUN_CONFIG_FILE="${RUN_CONFIG_FILE:-}"
if [[ -n "${RUN_CONFIG_FILE}" ]]; then
  RUN_CONFIG_FILE="$(realpath -m "${RUN_CONFIG_FILE}")"
  if [[ ! -f "${RUN_CONFIG_FILE}" ]]; then
    echo "[sps-watch] ERROR: RUN_CONFIG_FILE not found: ${RUN_CONFIG_FILE}"
    exit 1
  fi
  set -a
  # shellcheck disable=SC1090
  source "${RUN_CONFIG_FILE}"
  set +a
fi

SPS_THRESHOLD="${SPS_THRESHOLD:-7}"
SPS_CONSECUTIVE="${SPS_CONSECUTIVE:-3}"
SPS_GRACE_UPDATES="${SPS_GRACE_UPDATES:-5}"
MANAGE_CARLA="${MANAGE_CARLA:-1}"
CARLA_PORT="${CARLA_PORT:-2000}"
CARLA_BOOT_WAIT_SECONDS="${CARLA_BOOT_WAIT_SECONDS:-30}"
LEADERBOARD_READY_TIMEOUT_SECONDS="${LEADERBOARD_READY_TIMEOUT_SECONDS:-120}"
LEADERBOARD_READY_PATTERN="${LEADERBOARD_READY_PATTERN:-Connecting to gymnasium server}"
NO_PROGRESS_TIMEOUT_SECONDS="${NO_PROGRESS_TIMEOUT_SECONDS:-300}"
NO_PROGRESS_POLL_SECONDS="${NO_PROGRESS_POLL_SECONDS:-60}"

CHECKPOINT="$(realpath -m "${CHECKPOINT}")"
LOGDIR="$(realpath -m "${LOGDIR}")"
LOGFILE="${WATCHDOG_LOG:-${LOGDIR}/${EXP_NAME}/watchdog.log}"
HEARTBEAT_SECONDS="${WATCHDOG_HEARTBEAT_SECONDS:-60}"

mkdir -p "$(dirname "${LOGFILE}")"

run_pid=""
trainer_log=""
trigger_file=""
run_dir=""
sps_monitor_pid=""
stall_monitor_pid=""
heartbeat_pid=""
last_attached_run_dir=""

log() {
  local msg="$1"
  printf "[%s] [sps-watch] %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "${msg}" | tee -a "${LOGFILE}"
}

start_watchdog_heartbeat() {
  (
    while true; do
      sleep "${HEARTBEAT_SECONDS}"
      log "heartbeat watchdog_pid=$$ run_pid=${run_pid:-none} run_dir=${run_dir:-none}"
    done
  ) &
  heartbeat_pid=$!
}

on_exit() {
  local exit_code=$?
  if [[ -n "${heartbeat_pid}" ]]; then
    kill "${heartbeat_pid}" >/dev/null 2>&1 || true
  fi
  log "exit code=${exit_code} run_pid=${run_pid:-none} run_dir=${run_dir:-none} trigger_file=${trigger_file:-none}"
}

trap on_exit EXIT INT TERM
log "watchdog starting checkpoint=${CHECKPOINT} logdir=${LOGDIR} exp_name=${EXP_NAME} logfile=${LOGFILE}"
log "settings sps_threshold=${SPS_THRESHOLD} sps_consecutive=${SPS_CONSECUTIVE} no_progress_timeout=${NO_PROGRESS_TIMEOUT_SECONDS}s heartbeat=${HEARTBEAT_SECONDS}s"
if [[ -n "${RUN_CONFIG_FILE}" ]]; then
  log "run_config_file=${RUN_CONFIG_FILE}"
fi
start_watchdog_heartbeat

latest_checkpoint() {
  local exp_dir="$1"
  ls -t "${exp_dir}"/model_latest_*.pth 2>/dev/null | head -n 1 || true
}

start_carla() {
  if [[ "${MANAGE_CARLA}" == "1" ]]; then
    log "Starting CARLA on port ${CARLA_PORT}"
    "${REPO_ROOT}/scripts/start_carla.sh" "${CARLA_PORT}"
    log "Waiting ${CARLA_BOOT_WAIT_SECONDS}s for CARLA to be ready"
    sleep "${CARLA_BOOT_WAIT_SECONDS}"
  fi
}

stop_carla() {
  if [[ "${MANAGE_CARLA}" == "1" ]]; then
    log "Stopping CARLA"
    "${REPO_ROOT}/scripts/clean_carla.sh"
  fi
}

kill_leaderboard() {
  pkill -f "custom_leaderboard/leaderboard/leaderboard/leaderboard_evaluator.py" >/dev/null 2>&1 || true
}

kill_trainer() {
  pkill -f "train_tfv6_ppo.py" >/dev/null 2>&1 || true
  pkill -f "torchrun.*train_tfv6_ppo.py" >/dev/null 2>&1 || true
}

monitor_sps() {
  local trainer_log="$1"
  local run_pid="$2"
  local trigger_file="$3"

  local below=0
  local seen=0
  local last_count=0

  while true; do
    if ! kill -0 "${run_pid}" >/dev/null 2>&1; then
      exit 0
    fi

    if [[ -f "${trainer_log}" ]]; then
      local count
      count="$(grep -c "^SPS:" "${trainer_log}" 2>/dev/null || true)"
      count="${count:-0}"
      if (( count > last_count )); then
        local sps
        sps="$(awk '/^SPS:/{v=$2} END{print v}' "${trainer_log}" 2>/dev/null || true)"
        if [[ "${sps}" =~ ^[0-9]+$ ]]; then
          seen=$((seen + 1))
          if (( seen > SPS_GRACE_UPDATES )); then
            if (( sps < SPS_THRESHOLD )); then
              below=$((below + 1))
            else
              below=0
            fi
            if (( below >= SPS_CONSECUTIVE )); then
              log "SPS=${sps} below ${SPS_THRESHOLD} for ${below} updates; triggering restart."
              echo "SPS_TRIGGER ${sps}" >"${trigger_file}"
              if kill -0 "${run_pid}" >/dev/null 2>&1; then
                kill "${run_pid}" >/dev/null 2>&1 || true
              fi
              exit 2
            fi
          fi
        fi
        last_count="${count}"
      fi
    fi
    sleep 5
  done
}

wait_for_leaderboard_ready() {
  local log_file="$1"
  local deadline=$((SECONDS + LEADERBOARD_READY_TIMEOUT_SECONDS))
  while (( SECONDS < deadline )); do
    if [[ -f "${log_file}" ]] && grep -q "${LEADERBOARD_READY_PATTERN}" "${log_file}"; then
      return 0
    fi
    sleep 2
  done
  return 1
}

monitor_no_progress() {
  local log_file="$1"
  local run_pid="$2"
  local trigger_file="$3"
  while true; do
    if [[ -f "${log_file}" ]]; then
      local last_ts
      last_ts="$(stat -c %Y "${log_file}")"
      local now_ts
      now_ts="$(date +%s)"
      if (( now_ts - last_ts > NO_PROGRESS_TIMEOUT_SECONDS )); then
        log "No trainer log updates for ${NO_PROGRESS_TIMEOUT_SECONDS}s; triggering restart."
        echo "STALL_TRIGGER" >"${trigger_file}"
        if kill -0 "${run_pid}" >/dev/null 2>&1; then
          kill "${run_pid}" >/dev/null 2>&1 || true
        fi
        exit 2
      fi
    fi
    sleep "${NO_PROGRESS_POLL_SECONDS}"
  done
}

wait_for_fresh_run_dir() {
  local exp_dir="$1"
  local run_pid="$2"
  local timeout_seconds="$3"
  local deadline=$((SECONDS + timeout_seconds))
  while (( SECONDS < deadline )); do
    if ! kill -0 "${run_pid}" >/dev/null 2>&1; then
      return 1
    fi
    local candidate
    candidate="$(ls -td "${exp_dir}"/run_* 2>/dev/null | head -n 1 || true)"
    if [[ -n "${candidate}" && "${candidate}" != "${last_attached_run_dir}" && -f "${candidate}/trainer.log" ]]; then
      echo "${candidate}"
      return 0
    fi
    sleep 1
  done
  return 1
}

while true; do
  trigger_file=""
  sps_monitor_pid=""
  stall_monitor_pid=""
  start_carla

  exp_dir="${LOGDIR}/${EXP_NAME}"
  mkdir -p "${exp_dir}"
  resume_ckpt="$(latest_checkpoint "${exp_dir}")"

  if [[ -n "${resume_ckpt}" ]]; then
    log "Resuming from ${resume_ckpt}"
  else
    log "Starting fresh (no prior checkpoints)."
  fi

  CONTINUE_CHECKPOINT="${resume_ckpt}" \
    bash "${REPO_ROOT}/rl_finetuning/train_long_local.sh" \
      "${CHECKPOINT}" \
      "${LOGDIR}" \
      "${EXP_NAME}" &
  run_pid=$!

  # Wait for a fresh run dir and trainer log (avoid re-attaching stale runs).
  trainer_log=""
  run_dir=""
  run_dir="$(wait_for_fresh_run_dir "${exp_dir}" "${run_pid}" 180 || true)"
  if [[ -n "${run_dir}" ]]; then
    trainer_log="${run_dir}/trainer.log"
    trigger_file="${run_dir}/sps_triggered.txt"
    last_attached_run_dir="${run_dir}"
  fi

  if [[ -z "${trainer_log}" ]]; then
    log "No fresh trainer.log found within timeout; restarting."
    kill "${run_pid}" >/dev/null 2>&1 || true
    kill_leaderboard
    kill_trainer
    stop_carla
    sleep 5
    continue
  fi
  log "Attached to run_dir=${run_dir} trainer_log=${trainer_log}"

  if ! wait_for_leaderboard_ready "${run_dir}/leaderboard.log"; then
    log "Leaderboard not ready after ${LEADERBOARD_READY_TIMEOUT_SECONDS}s; restarting."
    kill_leaderboard
    kill_trainer
    stop_carla
    sleep 5
    continue
  fi

  monitor_sps "${trainer_log}" "${run_pid}" "${trigger_file}" &
  sps_monitor_pid=$!
  monitor_no_progress "${trainer_log}" "${run_pid}" "${trigger_file}" &
  stall_monitor_pid=$!

  run_status=0
  wait "${run_pid}" || run_status=$?
  kill "${sps_monitor_pid}" >/dev/null 2>&1 || true
  kill "${stall_monitor_pid}" >/dev/null 2>&1 || true

  if [[ "${run_status}" != "0" && ! -f "${trigger_file}" ]]; then
    log "Run exited with status ${run_status}; triggering restart."
    echo "RUN_EXIT ${run_status}" >"${trigger_file}"
  fi

  if [[ -f "${trigger_file}" ]]; then
    log "Restarting due to trigger: $(cat "${trigger_file}")"
    kill_leaderboard
    kill_trainer
    stop_carla
    sleep 5
    continue
  fi

  log "Training finished without trigger."
  break
done
