#!/usr/bin/env bash
set -euo pipefail

# Long local TFv6 PPO training run (leaderboard + trainer in one script).
# Designed as the single source of truth for long runs.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CARL_ROOT="${REPO_ROOT}/3rd_party/CaRL/CARLA"

CHECKPOINT="${1:-${REPO_ROOT}/outputs/checkpoints/tfv6_resnet34}"
LOGDIR="${2:-${REPO_ROOT}/outputs/rl_logs}"
EXP_NAME="${3:-TFV6_PPO_LONG_LOCAL}"
CONTINUE_CHECKPOINT="${CONTINUE_CHECKPOINT:-}"
shift_args=3
if [[ -n "${4:-}" && "${4}" != --* ]]; then
  CONTINUE_CHECKPOINT="${4}"
  shift_args=4
fi

TFV6_DEBUG_VIZ="${TFV6_DEBUG_VIZ:-0}"
TFV6_DEBUG_VIZ_EVERY_N="${TFV6_DEBUG_VIZ_EVERY_N:-1}"
TFV6_DEBUG_VIZ_MAX_IMAGES="${TFV6_DEBUG_VIZ_MAX_IMAGES:-0}"

if (( $# >= shift_args )); then
  shift "${shift_args}"
else
  shift "$#"
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --debug-viz)
      TFV6_DEBUG_VIZ=1
      shift
      ;;
    --debug-viz-every-n)
      TFV6_DEBUG_VIZ_EVERY_N="${2:?missing value for --debug-viz-every-n}"
      shift 2
      ;;
    --debug-viz-max-images)
      TFV6_DEBUG_VIZ_MAX_IMAGES="${2:?missing value for --debug-viz-max-images}"
      shift 2
      ;;
    *)
      echo "[train-long-local] ERROR: unknown argument '$1'"
      exit 1
      ;;
  esac
done

RUN_CONFIG_FILE="${RUN_CONFIG_FILE:-}"
if [[ -n "${RUN_CONFIG_FILE}" ]]; then
  RUN_CONFIG_FILE="$(realpath -m "${RUN_CONFIG_FILE}")"
  if [[ ! -f "${RUN_CONFIG_FILE}" ]]; then
    echo "[train-long-local] ERROR: RUN_CONFIG_FILE not found: ${RUN_CONFIG_FILE}"
    exit 1
  fi
  set -a
  # shellcheck disable=SC1090
  source "${RUN_CONFIG_FILE}"
  set +a
fi

GYM_PORT="${GYM_PORT:-5555}"
CARLA_PORT="${CARLA_PORT:-2000}"
TM_PORT="${TM_PORT:-8000}"
TCP_STORE_PORT="${TCP_STORE_PORT:-7000}"

TRACK="${TRACK:-MAP_QUALIFIER}"
ROUTE_PROFILE="${ROUTE_PROFILE:-town03_debug}"

# PPO defaults for long local runs.
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-256}"
TOTAL_MINIBATCH_SIZE="${TOTAL_MINIBATCH_SIZE:-64}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-3}"

# Runtime targeting via assumed SPS (can be overridden by TOTAL_TIMESTEPS).
TARGET_HOURS="${TARGET_HOURS:-12}"
ASSUMED_SPS="${ASSUMED_SPS:-5.0}"
if [[ -z "${TOTAL_TIMESTEPS:-}" ]]; then
  TOTAL_TIMESTEPS="$(python - <<PY
target_hours = float("${TARGET_HOURS}")
sps = float("${ASSUMED_SPS}")
print(int(target_hours * 3600.0 * sps))
PY
)"
fi

# Avoid route exhaustion by default.
REPETITIONS="${REPETITIONS:-500}"
LEADERBOARD_DEBUG="${LEADERBOARD_DEBUG:-0}"
HEARTBEAT_STEPS="${HEARTBEAT_STEPS:-64}"
DEBUG_MEMORY="${DEBUG_MEMORY:-0}"
GPU_TELEMETRY_ENABLE="${GPU_TELEMETRY_ENABLE:-1}"
GPU_TELEMETRY_INTERVAL_MS="${GPU_TELEMETRY_INTERVAL_MS:-500}"

CHECKPOINT="$(realpath -m "${CHECKPOINT}")"
LOGDIR="$(realpath -m "${LOGDIR}")"

if [[ ! -f "${CHECKPOINT}/config.json" ]]; then
  echo "[train-long-local] ERROR: ${CHECKPOINT}/config.json not found"
  exit 1
fi
if ! compgen -G "${CHECKPOINT}/model*.pth" >/dev/null; then
  echo "[train-long-local] ERROR: no model*.pth found in ${CHECKPOINT}"
  exit 1
fi

build_debug_suite_routes() {
  local output_file="$1"
  local towns_csv="${DEBUG_SUITE_TOWNS:-Town01,Town02,Town03,Town04,Town05,Town06}"
  python - "$CARL_ROOT" "$output_file" "$towns_csv" <<'PY'
import gzip
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

carl_root = Path(sys.argv[1])
out_file = Path(sys.argv[2])
towns = {x.strip() for x in sys.argv[3].split(",") if x.strip()}

src_dir = carl_root / "custom_leaderboard" / "leaderboard" / "data" / "debug_routes_with_scenarios"
files = sorted(src_dir.glob("route_Town*.xml.gz"))
if not files:
    raise FileNotFoundError(f"No route files found under {src_dir}")

root = ET.Element("routes")
route_id = 0
for file in files:
    town_name = file.name.replace("route_", "").replace("_00.xml.gz", "")
    if town_name not in towns:
        continue
    with gzip.open(file, "rt", encoding="utf-8") as f:
        tree = ET.parse(f)
    for route in tree.getroot().findall("route"):
        route.set("id", str(route_id))
        route_id += 1
        root.append(route)

if route_id == 0:
    raise RuntimeError(f"No routes selected. towns={sorted(towns)}")

ET.indent(root, space="  ")
xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
with gzip.open(out_file, "wb") as f:
    f.write(xml_bytes)
print(f"[train-long-local] Built merged route file: {out_file} ({route_id} routes)")
PY
}

if [[ "${ROUTE_PROFILE}" == "town03_debug" ]]; then
  ROUTE_FILE="${ROUTE_FILE:-${CARL_ROOT}/custom_leaderboard/leaderboard/data/debug_routes_with_scenarios/route_Town03_00.xml.gz}"
elif [[ "${ROUTE_PROFILE}" == "debug_suite" ]]; then
  ROUTE_FILE="${ROUTE_FILE:-/tmp/tfv6_debug_suite_routes.xml.gz}"
  build_debug_suite_routes "${ROUTE_FILE}"
elif [[ "${ROUTE_PROFILE}" == "training" ]]; then
  ROUTE_FILE="${ROUTE_FILE:-${CARL_ROOT}/custom_leaderboard/leaderboard/data/routes_training.xml}"
  REPETITIONS="${REPETITIONS:-1}"
else
  echo "[train-long-local] ERROR: unsupported ROUTE_PROFILE='${ROUTE_PROFILE}'"
  echo "Supported: town03_debug | debug_suite | training"
  exit 1
fi

ROUTE_FILE="$(realpath -m "${ROUTE_FILE}")"
if [[ ! -f "${ROUTE_FILE}" ]]; then
  echo "[train-long-local] ERROR: route file not found: ${ROUTE_FILE}"
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
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${LOGDIR}/${EXP_NAME}/run_${RUN_STAMP}"
mkdir -p "${RUN_DIR}"
LEADERBOARD_LOG="${RUN_DIR}/leaderboard.log"
TRAINER_LOG="${RUN_DIR}/trainer.log"
GPU_TELEMETRY_LOG="${RUN_DIR}/gpu_telemetry.csv"
GPU_APPS_LOG="${RUN_DIR}/gpu_processes.csv"
GPU_TELEMETRY_PID=""
GPU_APPS_PID=""

EST_HOURS="$(python - <<PY
steps = float("${TOTAL_TIMESTEPS}")
sps = float("${ASSUMED_SPS}")
print(f"{steps / sps / 3600.0:.2f}")
PY
)"

echo "[train-long-local] checkpoint=${CHECKPOINT}"
echo "[train-long-local] route_profile=${ROUTE_PROFILE}"
echo "[train-long-local] route_file=${ROUTE_FILE}"
echo "[train-long-local] repetitions=${REPETITIONS}"
echo "[train-long-local] total_timesteps=${TOTAL_TIMESTEPS}"
echo "[train-long-local] est_runtime_hours~=${EST_HOURS} (assumed_sps=${ASSUMED_SPS})"
echo "[train-long-local] total_batch_size=${TOTAL_BATCH_SIZE}"
echo "[train-long-local] total_minibatch_size=${TOTAL_MINIBATCH_SIZE}"
echo "[train-long-local] update_epochs=${UPDATE_EPOCHS}"
echo "[train-long-local] track=${TRACK}"
echo "[train-long-local] debug_memory=${DEBUG_MEMORY}"
echo "[train-long-local] debug_viz=${TFV6_DEBUG_VIZ}"
echo "[train-long-local] debug_viz_every_n=${TFV6_DEBUG_VIZ_EVERY_N}"
echo "[train-long-local] debug_viz_max_images=${TFV6_DEBUG_VIZ_MAX_IMAGES}"
echo "[train-long-local] logs_dir=${RUN_DIR}"
if [[ -n "${CONTINUE_CHECKPOINT}" ]]; then
  echo "[train-long-local] continue_checkpoint=${CONTINUE_CHECKPOINT}"
fi
if [[ -n "${RUN_CONFIG_FILE}" ]]; then
  echo "[train-long-local] run_config_file=${RUN_CONFIG_FILE}"
fi

EXTRA_TRAIN_ARGS=()
if [[ -n "${TRAINER_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_TRAIN_ARGS=(${TRAINER_EXTRA_ARGS})
  echo "[train-long-local] trainer_extra_args=${TRAINER_EXTRA_ARGS}"
fi

if [[ "${GPU_TELEMETRY_ENABLE}" == "1" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  INTERVAL_SECONDS="$(awk "BEGIN { printf \"%.3f\", ${GPU_TELEMETRY_INTERVAL_MS} / 1000 }")"
  nvidia-smi \
    --query-gpu=timestamp,index,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,power.draw \
    --format=csv \
    -lms "${GPU_TELEMETRY_INTERVAL_MS}" \
    >"${GPU_TELEMETRY_LOG}" 2>&1 &
  GPU_TELEMETRY_PID=$!
  (
    echo "timestamp,pid,process_name,used_gpu_memory_mib"
    while true; do
      ts="$(date '+%Y-%m-%d %H:%M:%S.%3N')"
      nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory \
        --format=csv,noheader,nounits 2>/dev/null \
        | awk -v t="${ts}" 'NF { print t "," $0 }'
      sleep "${INTERVAL_SECONDS}"
    done
  ) >"${GPU_APPS_LOG}" 2>&1 &
  GPU_APPS_PID=$!
  echo "[train-long-local] gpu_telemetry=${GPU_TELEMETRY_LOG}"
  echo "[train-long-local] gpu_processes=${GPU_APPS_LOG}"
else
  echo "[train-long-local] GPU telemetry disabled (set GPU_TELEMETRY_ENABLE=1 and ensure nvidia-smi is available)."
fi

echo "[train-long-local] Starting leaderboard client (expects CARLA server on port ${CARLA_PORT})"
python -u "${CARL_ROOT}/custom_leaderboard/leaderboard/leaderboard/leaderboard_evaluator.py" \
  --host 127.0.0.1 \
  --port "${CARLA_PORT}" \
  --traffic-manager-port "${TM_PORT}" \
  --routes "${ROUTE_FILE}" \
  --repetitions "${REPETITIONS}" \
  --track "${TRACK}" \
  --agent "${REPO_ROOT}/rl_finetuning/tfv6_rl/env_agent_tfv6.py" \
  --agent-config "${CHECKPOINT}" \
  --gym_port "${GYM_PORT}" \
  --checkpoint "${LOGDIR}/${EXP_NAME}/route_0.json" \
  --resume 0 \
  --debug "${LEADERBOARD_DEBUG}" \
  --frame_rate 10 \
  --timeout 900 \
  --runtime_timeout 900 \
  --no_rendering_mode True \
  >"${LEADERBOARD_LOG}" 2>&1 &
LEADERBOARD_PID=$!

cleanup() {
  if kill -0 "${LEADERBOARD_PID}" >/dev/null 2>&1; then
    kill "${LEADERBOARD_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${GPU_TELEMETRY_PID}" ]] && kill -0 "${GPU_TELEMETRY_PID}" >/dev/null 2>&1; then
    kill "${GPU_TELEMETRY_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${GPU_APPS_PID}" ]] && kill -0 "${GPU_APPS_PID}" >/dev/null 2>&1; then
    kill "${GPU_APPS_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

sleep 2

echo "[train-long-local] Starting TFv6 PPO trainer"
LOAD_ARG=()
if [[ -n "${CONTINUE_CHECKPOINT}" ]]; then
  CONTINUE_CHECKPOINT="$(realpath -m "${CONTINUE_CHECKPOINT}")"
  if [[ ! -f "${CONTINUE_CHECKPOINT}" ]]; then
    echo "[train-long-local] ERROR: continue checkpoint not found: ${CONTINUE_CHECKPOINT}"
    exit 1
  fi
  LOAD_ARG=(--load_file "${CONTINUE_CHECKPOINT}")
fi

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
  --heartbeat_steps "${HEARTBEAT_STEPS}" \
  --debug_memory "${DEBUG_MEMORY}" \
  --run_dir "${RUN_DIR}" \
  --debug_viz "${TFV6_DEBUG_VIZ}" \
  --debug_viz_every_n "${TFV6_DEBUG_VIZ_EVERY_N}" \
  --debug_viz_max_images "${TFV6_DEBUG_VIZ_MAX_IMAGES}" \
  "${EXTRA_TRAIN_ARGS[@]}" \
  "${LOAD_ARG[@]}" \
  2>&1 | tee "${TRAINER_LOG}"

echo "[train-long-local] Done. Check TensorBoard logs at ${LOGDIR}/${EXP_NAME}"
echo "[train-long-local] Leaderboard log: ${LEADERBOARD_LOG}"
echo "[train-long-local] Trainer log: ${TRAINER_LOG}"
