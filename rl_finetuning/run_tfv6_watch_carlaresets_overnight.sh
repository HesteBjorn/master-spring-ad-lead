#!/usr/bin/env bash
set -euo pipefail

# SPS watchdog for long overnight runs.
# Restarts CARLA + trainer when SPS falls below a threshold.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CHECKPOINT="${1:-${REPO_ROOT}/outputs/checkpoints/tfv6_resnet34}"
LOGDIR="${2:-${REPO_ROOT}/outputs/rl_logs}"
EXP_NAME="${3:-TFV6_PPO_SMOKE_OVERNIGHT}"

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

latest_checkpoint() {
  local exp_dir="$1"
  ls -t "${exp_dir}"/model_latest_*.pth 2>/dev/null | head -n 1 || true
}

start_carla() {
  if [[ "${MANAGE_CARLA}" == "1" ]]; then
    echo "[sps-watch] Starting CARLA on port ${CARLA_PORT}"
    "${REPO_ROOT}/scripts/start_carla.sh" "${CARLA_PORT}"
    echo "[sps-watch] Waiting ${CARLA_BOOT_WAIT_SECONDS}s for CARLA to be ready"
    sleep "${CARLA_BOOT_WAIT_SECONDS}"
  fi
}

stop_carla() {
  if [[ "${MANAGE_CARLA}" == "1" ]]; then
    echo "[sps-watch] Stopping CARLA"
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

  tail -n 0 -F "${trainer_log}" | while read -r line; do
    if ! kill -0 "${run_pid}" >/dev/null 2>&1; then
      exit 0
    fi
    if [[ "${line}" =~ SPS:\ ([0-9]+) ]]; then
      local sps="${BASH_REMATCH[1]}"
      seen=$((seen + 1))
      if (( seen <= SPS_GRACE_UPDATES )); then
        continue
      fi
      if (( sps < SPS_THRESHOLD )); then
        below=$((below + 1))
      else
        below=0
      fi
      if (( below >= SPS_CONSECUTIVE )); then
        echo "[sps-watch] SPS=${sps} below ${SPS_THRESHOLD} for ${below} updates; triggering restart."
        echo "SPS_TRIGGER ${sps}" >"${trigger_file}"
        kill "${run_pid}" >/dev/null 2>&1 || true
        exit 2
      fi
    fi
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
        echo "[sps-watch] No trainer log updates for ${NO_PROGRESS_TIMEOUT_SECONDS}s; triggering restart."
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

while true; do
  start_carla

  exp_dir="${LOGDIR}/${EXP_NAME}"
  mkdir -p "${exp_dir}"
  resume_ckpt="$(latest_checkpoint "${exp_dir}")"

  if [[ -n "${resume_ckpt}" ]]; then
    echo "[sps-watch] Resuming from ${resume_ckpt}"
  else
    echo "[sps-watch] Starting fresh (no prior checkpoints)."
  fi

  CONTINUE_CHECKPOINT="${resume_ckpt}" \
    bash "${REPO_ROOT}/rl_finetuning/run_tfv6_smoke_overnight.sh" \
      "${CHECKPOINT}" \
      "${LOGDIR}" \
      "${EXP_NAME}" &
  run_pid=$!

  # Wait for trainer log.
  trainer_log=""
  for _ in {1..120}; do
    run_dir="$(ls -td "${exp_dir}"/run_* 2>/dev/null | head -n 1 || true)"
    if [[ -n "${run_dir}" && -f "${run_dir}/trainer.log" ]]; then
      trainer_log="${run_dir}/trainer.log"
      trigger_file="${run_dir}/sps_triggered.txt"
      break
    fi
    sleep 1
  done

  if [[ -z "${trainer_log}" ]]; then
    echo "[sps-watch] ERROR: trainer.log not found; aborting."
    kill "${run_pid}" >/dev/null 2>&1 || true
    stop_carla
    exit 1
  fi

  if ! wait_for_leaderboard_ready "${run_dir}/leaderboard.log"; then
    echo "[sps-watch] Leaderboard not ready after ${LEADERBOARD_READY_TIMEOUT_SECONDS}s; restarting."
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
    echo "[sps-watch] Run exited with status ${run_status}; triggering restart."
    echo "RUN_EXIT ${run_status}" >"${trigger_file}"
  fi

  if [[ -f "${trigger_file}" ]]; then
    echo "[sps-watch] Restarting due to SPS trigger."
    kill_leaderboard
    kill_trainer
    stop_carla
    sleep 5
    continue
  fi

  echo "[sps-watch] Training finished without SPS trigger."
  break
done
