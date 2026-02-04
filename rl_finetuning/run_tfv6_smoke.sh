#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CARL_ROOT="${REPO_ROOT}/3rd_party/CaRL/CARLA"

CHECKPOINT="${1:-${REPO_ROOT}/outputs/checkpoints/tfv6_resnet34}"
LOGDIR="${2:-${REPO_ROOT}/outputs/rl_logs}"
EXP_NAME="${3:-TFV6_PPO_SMOKE}"

GYM_PORT="${GYM_PORT:-5555}"
CARLA_PORT="${CARLA_PORT:-2000}"
TM_PORT="${TM_PORT:-8000}"
TCP_STORE_PORT="${TCP_STORE_PORT:-7000}"
TRACK="${TRACK:-MAP_QUALIFIER}"

TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-32}"
TOTAL_MINIBATCH_SIZE="${TOTAL_MINIBATCH_SIZE:-32}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-32}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-1}"

ROUTE_FILE="${ROUTE_FILE:-${CARL_ROOT}/custom_leaderboard/leaderboard/data/debug_routes_with_scenarios/route_Town03_00.xml.gz}"

CHECKPOINT="$(realpath -m "${CHECKPOINT}")"
LOGDIR="$(realpath -m "${LOGDIR}")"
ROUTE_FILE="$(realpath -m "${ROUTE_FILE}")"

if [[ ! -f "${CHECKPOINT}/config.json" ]]; then
  echo "[smoke] ERROR: ${CHECKPOINT}/config.json not found"
  exit 1
fi

if ! compgen -G "${CHECKPOINT}/model*.pth" >/dev/null; then
  echo "[smoke] ERROR: no model*.pth found in ${CHECKPOINT}"
  exit 1
fi

export WORK_DIR="${CARL_ROOT}"
export SCENARIO_RUNNER_ROOT="${CARL_ROOT}/custom_leaderboard/scenario_runner"
export LEADERBOARD_ROOT="${CARL_ROOT}/custom_leaderboard/leaderboard"
export PYTHONPATH="${SCENARIO_RUNNER_ROOT}:${LEADERBOARD_ROOT}:${PYTHONPATH:-}"
export TFV6_RL_TRACK="${TRACK}"

if [[ -n "${CARLA_ROOT:-}" ]]; then
  export PYTHONPATH="${CARLA_ROOT}/PythonAPI:${CARLA_ROOT}/PythonAPI/carla:${PYTHONPATH}"
fi

mkdir -p "${LOGDIR}/${EXP_NAME}"

echo "[smoke] Starting leaderboard client (expects CARLA server already running on port ${CARLA_PORT})"
python -u "${CARL_ROOT}/custom_leaderboard/leaderboard/leaderboard/leaderboard_evaluator.py" \
  --host 127.0.0.1 \
  --port "${CARLA_PORT}" \
  --traffic-manager-port "${TM_PORT}" \
  --routes "${ROUTE_FILE}" \
  --repetitions 1 \
  --track "${TRACK}" \
  --agent "${REPO_ROOT}/rl_finetuning/tfv6_rl/env_agent_tfv6.py" \
  --agent-config "${CHECKPOINT}" \
  --gym_port "${GYM_PORT}" \
  --checkpoint "${LOGDIR}/${EXP_NAME}/route_0.json" \
  --resume 0 \
  --frame_rate 10 \
  --timeout 900 \
  --runtime_timeout 900 \
  --no_rendering_mode True &
LEADERBOARD_PID=$!

cleanup() {
  if kill -0 "${LEADERBOARD_PID}" >/dev/null 2>&1; then
    kill "${LEADERBOARD_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

sleep 2

echo "[smoke] Starting TFv6 PPO trainer"
torchrun --nnodes=1 --nproc_per_node=1 --max_restarts=0 \
  --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
  "${REPO_ROOT}/rl_finetuning/train_tfv6_ppo.py" \
  --tcp_store_port "${TCP_STORE_PORT}" \
  --logdir "${LOGDIR}" \
  --exp_name "${EXP_NAME}" \
  --ports "${GYM_PORT}" \
  --total_batch_size "${TOTAL_BATCH_SIZE}" \
  --total_minibatch_size "${TOTAL_MINIBATCH_SIZE}" \
  --update_epochs "${UPDATE_EPOCHS}" \
  --total_timesteps "${TOTAL_TIMESTEPS}" \
  --reward_type simple_reward \
  --tfv6_checkpoint "${CHECKPOINT}" \
  --debug_shapes 1 \
  --heartbeat_steps 8

echo "[smoke] Done. Check TensorBoard logs at ${LOGDIR}/${EXP_NAME}"
