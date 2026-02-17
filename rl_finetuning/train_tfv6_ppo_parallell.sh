#!/usr/bin/env bash
set -euo pipefail

# Convenient CaRL-style launcher for TFv6 PPO parallel training.
# Starts CARLA servers, leaderboard clients and torchrun via train_parallel_tfv6_ppo.py.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CARL_ROOT="${REPO_ROOT}/3rd_party/CaRL/CARLA"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  bash rl_finetuning/train_tfv6_ppo_parallell.sh [CHECKPOINT] [LOGDIR] [EXP_NAME] [extra trainer args...]

Description:
  CaRL-style robust parallel TFv6 PPO launcher. Starts CARLA servers, leaderboard clients,
  and distributed trainer workers with auto-restart on crashes.

Examples:
  bash rl_finetuning/train_tfv6_ppo_parallell.sh \
    outputs/checkpoints/tfv6_resnet34 \
    outputs/rl_logs \
    TFV6_PPO_PARALLEL

  RUN_CONFIG_FILE=rl_finetuning/configs/weekend_local_run_stable.env \
  NUM_ENVS_PER_NODE=4 NUM_ENVS_PER_GPU=1 GPU_IDS="0 1 2 3" \
  bash rl_finetuning/train_tfv6_ppo_parallell.sh \
    outputs/checkpoints/tfv6_resnet34 \
    outputs/rl_logs \
    TFV6_PPO_PARALLEL_MULTI
EOF
  exit 0
fi

CHECKPOINT="${1:-${REPO_ROOT}/outputs/checkpoints/tfv6_resnet34}"
LOGDIR="${2:-${REPO_ROOT}/outputs/rl_logs}"
EXP_NAME="${3:-TFV6_PPO_PARALLEL}"
shift_args=3
if (( $# >= shift_args )); then
  shift "${shift_args}"
else
  shift "$#"
fi

RUN_CONFIG_FILE="${RUN_CONFIG_FILE:-}"
if [[ -n "${RUN_CONFIG_FILE}" ]]; then
  RUN_CONFIG_FILE="$(realpath -m "${RUN_CONFIG_FILE}")"
  if [[ ! -f "${RUN_CONFIG_FILE}" ]]; then
    echo "[train-tfv6-ppo-parallell] ERROR: RUN_CONFIG_FILE not found: ${RUN_CONFIG_FILE}"
    exit 1
  fi
  set -a
  # shellcheck disable=SC1090
  source "${RUN_CONFIG_FILE}"
  set +a
fi

CHECKPOINT="$(realpath -m "${CHECKPOINT}")"
LOGDIR="$(realpath -m "${LOGDIR}")"
CARLA_ROOT="${CARLA_ROOT:-${REPO_ROOT}/3rd_party/CARLA_0915}"
CARLA_ROOT="$(realpath -m "${CARLA_ROOT}")"

if [[ ! -d "${CHECKPOINT}" ]]; then
  echo "[train-tfv6-ppo-parallell] ERROR: checkpoint directory not found: ${CHECKPOINT}"
  exit 1
fi
if [[ ! -f "${CHECKPOINT}/config.json" ]]; then
  echo "[train-tfv6-ppo-parallell] ERROR: missing ${CHECKPOINT}/config.json"
  exit 1
fi
if ! compgen -G "${CHECKPOINT}/model*.pth" >/dev/null; then
  echo "[train-tfv6-ppo-parallell] ERROR: no model*.pth found in ${CHECKPOINT}"
  exit 1
fi
if [[ ! -x "${CARLA_ROOT}/CarlaUE4.sh" ]]; then
  echo "[train-tfv6-ppo-parallell] ERROR: CARLA launcher not found at ${CARLA_ROOT}/CarlaUE4.sh"
  exit 1
fi

NUM_ENVS_PER_NODE="${NUM_ENVS_PER_NODE:-1}"
NUM_ENVS_PER_GPU="${NUM_ENVS_PER_GPU:-1}"
NUM_NODES="${NUM_NODES:-1}"
NODE_ID="${NODE_ID:-0}"
RDZV_ADDR="${RDZV_ADDR:-127.0.0.1}"
RDZV_PORT="${RDZV_PORT:-29500}"
START_PORT="${START_PORT:-1024}"
SEED="${SEED:-0}"
TRAIN_TOWNS="${TRAIN_TOWNS:-3}"
ROUTES_FOLDER="${ROUTES_FOLDER:-debug_routes_with_scenarios}"
ROUTE_REPETITIONS="${ROUTE_REPETITIONS:-500}"
TRACK="${TRACK:-MAP_QUALIFIER}"
FRAME_RATE="${FRAME_RATE:-10.0}"
TIMEOUT="${TIMEOUT:-900.0}"
RUNTIME_TIMEOUT="${RUNTIME_TIMEOUT:-900.0}"
LEADERBOARD_DEBUG="${LEADERBOARD_DEBUG:-0}"
ML_CLOUD="${ML_CLOUD:-0}"
USE_TRAJ_SYNC_PPO="${USE_TRAJ_SYNC_PPO:-0}"
GPU_IDS="${GPU_IDS:-0}"
CARLA_RPC_THREADS="${CARLA_RPC_THREADS:-8}"
CARLA_STREAMING_THREADS="${CARLA_STREAMING_THREADS:-8}"
CARLA_SECONDARY_THREADS="${CARLA_SECONDARY_THREADS:-8}"

TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-256}"
TOTAL_MINIBATCH_SIZE="${TOTAL_MINIBATCH_SIZE:-64}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-3}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-10000000}"

EXTRA_TRAIN_ARGS=()
if [[ -n "${TRAINER_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_TRAIN_ARGS=(${TRAINER_EXTRA_ARGS})
fi

echo "[train-tfv6-ppo-parallell] checkpoint=${CHECKPOINT}"
echo "[train-tfv6-ppo-parallell] logdir=${LOGDIR}"
echo "[train-tfv6-ppo-parallell] exp_name=${EXP_NAME}"
echo "[train-tfv6-ppo-parallell] carla_root=${CARLA_ROOT}"
echo "[train-tfv6-ppo-parallell] num_envs_per_node=${NUM_ENVS_PER_NODE} num_envs_per_gpu=${NUM_ENVS_PER_GPU}"
echo "[train-tfv6-ppo-parallell] num_nodes=${NUM_NODES} node_id=${NODE_ID} rdzv=${RDZV_ADDR}:${RDZV_PORT}"
echo "[train-tfv6-ppo-parallell] train_towns=${TRAIN_TOWNS} routes_folder=${ROUTES_FOLDER}"
echo "[train-tfv6-ppo-parallell] carla_server_profile=tfv6_sensor_safe (threading_on, no_rendering_mode=False)"
echo "[train-tfv6-ppo-parallell] carla_threads rpc=${CARLA_RPC_THREADS} streaming=${CARLA_STREAMING_THREADS} secondary=${CARLA_SECONDARY_THREADS}"
echo "[train-tfv6-ppo-parallell] total_batch_size=${TOTAL_BATCH_SIZE} total_minibatch_size=${TOTAL_MINIBATCH_SIZE} update_epochs=${UPDATE_EPOCHS} total_timesteps=${TOTAL_TIMESTEPS}"

python -u "${REPO_ROOT}/rl_finetuning/train_parallel_tfv6_ppo.py" \
  --repo_root "${REPO_ROOT}" \
  --git_root "${CARL_ROOT}" \
  --carla_root "${CARLA_ROOT}" \
  --tfv6_checkpoint "${CHECKPOINT}" \
  --log_root "${LOGDIR}" \
  --exp_name "${EXP_NAME}" \
  --num_envs_per_node "${NUM_ENVS_PER_NODE}" \
  --num_envs_per_gpu "${NUM_ENVS_PER_GPU}" \
  --num_nodes "${NUM_NODES}" \
  --node_id "${NODE_ID}" \
  --rdzv_addr "${RDZV_ADDR}" \
  --rdzv_port "${RDZV_PORT}" \
  --start_port "${START_PORT}" \
  --seed "${SEED}" \
  --train_towns ${TRAIN_TOWNS} \
  --routes_folder "${ROUTES_FOLDER}" \
  --route_repetitions "${ROUTE_REPETITIONS}" \
  --track "${TRACK}" \
  --frame_rate "${FRAME_RATE}" \
  --timeout "${TIMEOUT}" \
  --runtime_timeout "${RUNTIME_TIMEOUT}" \
  --carla_rpc_threads "${CARLA_RPC_THREADS}" \
  --carla_streaming_threads "${CARLA_STREAMING_THREADS}" \
  --carla_secondary_threads "${CARLA_SECONDARY_THREADS}" \
  --leaderboard_debug "${LEADERBOARD_DEBUG}" \
  --ml_cloud "${ML_CLOUD}" \
  --use_traj_sync_ppo "${USE_TRAJ_SYNC_PPO}" \
  --gpu_ids ${GPU_IDS} \
  --total_batch_size "${TOTAL_BATCH_SIZE}" \
  --total_minibatch_size "${TOTAL_MINIBATCH_SIZE}" \
  --update_epochs "${UPDATE_EPOCHS}" \
  --total_timesteps "${TOTAL_TIMESTEPS}" \
  --reward_type simple_reward \
  "${EXTRA_TRAIN_ARGS[@]}" \
  "$@"
