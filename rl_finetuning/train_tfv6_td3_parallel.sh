#!/usr/bin/env bash
set -euo pipefail

# CaRL-style launcher for TFv6 TD3 parallel training.
# Starts CARLA servers, leaderboard clients, and single-process TD3 training.
# Mirrors train_tfv6_ppo_parallell.sh — only TD3-specific env vars differ.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CARL_ROOT="${REPO_ROOT}/3rd_party/CaRL/CARLA"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  bash rl_finetuning/train_tfv6_td3_parallel.sh [CHECKPOINT] [LOGDIR] [EXP_NAME] [extra trainer args...]

Description:
  CaRL-style robust parallel TFv6 TD3 launcher. Starts CARLA servers, leaderboard clients,
  and single-process TD3 training with auto-restart on crashes.

Examples:
  RUN_CONFIG_FILE=rl_finetuning/configs/train_local_residual_td3.env \
  bash rl_finetuning/train_tfv6_td3_parallel.sh \
    outputs/checkpoints/tfv6_resnet34 \
    outputs/rl_logs \
    TFV6_TD3_LOCAL_RESIDUAL_smoke
EOF
  exit 0
fi

CHECKPOINT="${1:-${REPO_ROOT}/outputs/checkpoints/tfv6_resnet34}"
LOGDIR="${2:-${REPO_ROOT}/outputs/rl_logs}"
EXP_NAME="${3:-TFV6_TD3_PARALLEL}"
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
    echo "[train-tfv6-td3-parallel] ERROR: RUN_CONFIG_FILE not found: ${RUN_CONFIG_FILE}"
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
  echo "[train-tfv6-td3-parallel] ERROR: checkpoint directory not found: ${CHECKPOINT}"
  exit 1
fi
if [[ ! -f "${CHECKPOINT}/config.json" ]]; then
  echo "[train-tfv6-td3-parallel] ERROR: missing ${CHECKPOINT}/config.json"
  exit 1
fi
if ! compgen -G "${CHECKPOINT}/model*.pth" >/dev/null; then
  echo "[train-tfv6-td3-parallel] ERROR: no model*.pth found in ${CHECKPOINT}"
  exit 1
fi
if [[ ! -x "${CARLA_ROOT}/CarlaUE4.sh" ]]; then
  echo "[train-tfv6-td3-parallel] ERROR: CARLA launcher not found at ${CARLA_ROOT}/CarlaUE4.sh"
  exit 1
fi

NUM_ENVS_PER_NODE="${NUM_ENVS_PER_NODE:-1}"
NUM_ENVS_PER_GPU="${NUM_ENVS_PER_GPU:-1}"
NUM_NODES="${NUM_NODES:-1}"
NODE_ID="${NODE_ID:-0}"
START_PORT="${START_PORT:-1024}"
SEED="${SEED:-0}"
TRAIN_TOWNS="${TRAIN_TOWNS:-3}"
ROUTES_FOLDER="${ROUTES_FOLDER:-3rd_party/CaRL/CARLA/custom_leaderboard/leaderboard/data/debug_routes_with_scenarios}"
ROUTE_REPETITIONS="${ROUTE_REPETITIONS:-500}"
TRACK="${TRACK:-MAP_QUALIFIER}"
FRAME_RATE="${FRAME_RATE:-10.0}"
TIMEOUT="${TIMEOUT:-900.0}"
RUNTIME_TIMEOUT="${RUNTIME_TIMEOUT:-900.0}"
LEADERBOARD_DEBUG="${LEADERBOARD_DEBUG:-0}"
ML_CLOUD="${ML_CLOUD:-0}"
GPU_IDS="${GPU_IDS:-0}"
CARLA_RPC_THREADS="${CARLA_RPC_THREADS:-8}"
CARLA_STREAMING_THREADS="${CARLA_STREAMING_THREADS:-8}"
CARLA_SECONDARY_THREADS="${CARLA_SECONDARY_THREADS:-8}"

# TD3-specific replay-buffer / update parameters.
BUFFER_SIZE="${BUFFER_SIZE:-100000}"
LEARNING_STARTS="${LEARNING_STARTS:-5000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-10000000}"
TAU="${TAU:-0.005}"
UTD_ACTOR="${UTD_ACTOR:-0.5}"
EXPLORATION_NOISE="${EXPLORATION_NOISE:-0.1}"
TARGET_POLICY_NOISE="${TARGET_POLICY_NOISE:-0.2}"
TARGET_NOISE_CLIP="${TARGET_NOISE_CLIP:-0.5}"
CRITIC_WARMUP_STEPS="${CRITIC_WARMUP_STEPS:-10000}"
ACTOR_LR="${ACTOR_LR:-1e-4}"
CRITIC_LR="${CRITIC_LR:-1e-3}"
SAVE_EVERY="${SAVE_EVERY:-10000}"
UTD_RATIO="${UTD_RATIO:-1}"

EXTRA_TRAIN_ARGS=()
if [[ -n "${TRAINER_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_TRAIN_ARGS=(${TRAINER_EXTRA_ARGS})
fi

echo "[train-tfv6-td3-parallel] checkpoint=${CHECKPOINT}"
echo "[train-tfv6-td3-parallel] logdir=${LOGDIR}"
echo "[train-tfv6-td3-parallel] exp_name=${EXP_NAME}"
echo "[train-tfv6-td3-parallel] carla_root=${CARLA_ROOT}"
echo "[train-tfv6-td3-parallel] num_envs_per_node=${NUM_ENVS_PER_NODE} num_envs_per_gpu=${NUM_ENVS_PER_GPU}"
echo "[train-tfv6-td3-parallel] buffer_size=${BUFFER_SIZE} batch_size=${BATCH_SIZE} learning_starts=${LEARNING_STARTS} total_timesteps=${TOTAL_TIMESTEPS}"
echo "[train-tfv6-td3-parallel] tau=${TAU} utd_ratio=${UTD_RATIO} utd_actor=${UTD_ACTOR} critic_warmup_steps=${CRITIC_WARMUP_STEPS}"

python -u "${REPO_ROOT}/rl_finetuning/train_parallel_tfv6_td3.py" \
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
  --gpu_ids ${GPU_IDS} \
  --buffer_size "${BUFFER_SIZE}" \
  --learning_starts "${LEARNING_STARTS}" \
  --td3_batch_size "${BATCH_SIZE}" \
  --total_timesteps "${TOTAL_TIMESTEPS}" \
  --tau "${TAU}" \
  --utd_actor "${UTD_ACTOR}" \
  --exploration_noise "${EXPLORATION_NOISE}" \
  --target_policy_noise "${TARGET_POLICY_NOISE}" \
  --target_noise_clip "${TARGET_NOISE_CLIP}" \
  --critic_warmup_steps "${CRITIC_WARMUP_STEPS}" \
  --actor_lr "${ACTOR_LR}" \
  --critic_lr "${CRITIC_LR}" \
  --save_every "${SAVE_EVERY}" \
  --utd_ratio "${UTD_RATIO}" \
  --reward_type simple_reward \
  "${EXTRA_TRAIN_ARGS[@]}" \
  "$@"
