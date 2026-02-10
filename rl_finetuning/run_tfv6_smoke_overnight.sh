#!/usr/bin/env bash
set -euo pipefail

# Overnight TFv6 PPO debug run.
# This wraps run_tfv6_smoke_long.sh with defaults tuned for a ~12h run.
# Rollout length matches CaRL v1.1 in env-steps-per-env-per-update (256).

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CHECKPOINT="${1:-${REPO_ROOT}/outputs/checkpoints/tfv6_resnet34}"
LOGDIR="${2:-${REPO_ROOT}/outputs/rl_logs}"
EXP_NAME="${3:-TFV6_PPO_SMOKE_OVERNIGHT}"
CONTINUE_CHECKPOINT="${4:-${CONTINUE_CHECKPOINT:-}}"

RUN_CONFIG_FILE="${RUN_CONFIG_FILE:-}"
if [[ -n "${RUN_CONFIG_FILE}" ]]; then
  RUN_CONFIG_FILE="$(realpath -m "${RUN_CONFIG_FILE}")"
  if [[ ! -f "${RUN_CONFIG_FILE}" ]]; then
    echo "[smoke-overnight] ERROR: RUN_CONFIG_FILE not found: ${RUN_CONFIG_FILE}"
    exit 1
  fi
  set -a
  # shellcheck disable=SC1090
  source "${RUN_CONFIG_FILE}"
  set +a
fi

# CaRL v1.1-style rollout length per env: 256 steps/update.
# With one environment, TOTAL_BATCH_SIZE is the per-update rollout length.
export TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-256}"
export TOTAL_MINIBATCH_SIZE="${TOTAL_MINIBATCH_SIZE:-64}"
export UPDATE_EPOCHS="${UPDATE_EPOCHS:-3}"

# For a 12h overnight run, set total timesteps from an assumed SPS.
# Override either TARGET_HOURS, ASSUMED_SPS or TOTAL_TIMESTEPS directly.
TARGET_HOURS="${TARGET_HOURS:-12}"
ASSUMED_SPS="${ASSUMED_SPS:-5.0}"
if [[ -z "${TOTAL_TIMESTEPS:-}" ]]; then
  export TOTAL_TIMESTEPS="$(python - <<PY
target_hours = float("${TARGET_HOURS}")
sps = float("${ASSUMED_SPS}")
print(int(target_hours * 3600.0 * sps))
PY
)"
else
  export TOTAL_TIMESTEPS
fi

# Avoid route exhaustion for long runs: 10 routes * 500 reps = 5000 routes.
export ROUTE_PROFILE="${ROUTE_PROFILE:-town03_debug}"
export REPETITIONS="${REPETITIONS:-500}"

export HEARTBEAT_STEPS="${HEARTBEAT_STEPS:-64}"
export DEBUG_MEMORY="${DEBUG_MEMORY:-0}"
export GPU_TELEMETRY_ENABLE="${GPU_TELEMETRY_ENABLE:-1}"
export GPU_TELEMETRY_INTERVAL_MS="${GPU_TELEMETRY_INTERVAL_MS:-500}"

EST_HOURS="$(python - <<PY
steps = float("${TOTAL_TIMESTEPS}")
sps = float("${ASSUMED_SPS}")
print(f"{steps / sps / 3600.0:.2f}")
PY
)"

echo "[smoke-overnight] rollout_steps_per_env_update=${TOTAL_BATCH_SIZE} (target: 256)"
echo "[smoke-overnight] total_timesteps=${TOTAL_TIMESTEPS}"
echo "[smoke-overnight] estimated_runtime_hours~=${EST_HOURS} (assumed_sps=${ASSUMED_SPS})"
echo "[smoke-overnight] repetitions=${REPETITIONS} route_profile=${ROUTE_PROFILE}"
if [[ -n "${CONTINUE_CHECKPOINT}" ]]; then
  echo "[smoke-overnight] continue_checkpoint=${CONTINUE_CHECKPOINT}"
fi
if [[ -n "${RUN_CONFIG_FILE}" ]]; then
  echo "[smoke-overnight] run_config_file=${RUN_CONFIG_FILE}"
fi

exec bash "${REPO_ROOT}/rl_finetuning/run_tfv6_smoke_long.sh" \
  "${CHECKPOINT}" \
  "${LOGDIR}" \
  "${EXP_NAME}" \
  "${CONTINUE_CHECKPOINT}"
