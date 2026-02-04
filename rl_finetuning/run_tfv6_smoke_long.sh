#!/usr/bin/env bash
set -euo pipefail

# Longer TFv6 PPO smoke/debug run.
# Keeps the CaRL two-process debug pattern (leaderboard + learner) but uses
# longer rollouts and more PPO updates for TensorBoard sanity checks.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CARL_ROOT="${REPO_ROOT}/3rd_party/CaRL/CARLA"

CHECKPOINT="${1:-${REPO_ROOT}/outputs/checkpoints/tfv6_resnet34}"
LOGDIR="${2:-${REPO_ROOT}/outputs/rl_logs}"
EXP_NAME="${3:-TFV6_PPO_SMOKE_LONG}"

GYM_PORT="${GYM_PORT:-5555}"
CARLA_PORT="${CARLA_PORT:-2000}"
TM_PORT="${TM_PORT:-8000}"
TCP_STORE_PORT="${TCP_STORE_PORT:-7000}"

TRACK="${TRACK:-MAP_QUALIFIER}"
ROUTE_PROFILE="${ROUTE_PROFILE:-town03_debug}"

# "About one hour" defaults on a single env/GPU (depends on SPS on your machine).
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-1024}"
TOTAL_MINIBATCH_SIZE="${TOTAL_MINIBATCH_SIZE:-256}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-30000}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-3}"
HEARTBEAT_STEPS="${HEARTBEAT_STEPS:-128}"
REPETITIONS="${REPETITIONS:-50}"
LEADERBOARD_DEBUG="${LEADERBOARD_DEBUG:-0}"
DEBUG_MEMORY="${DEBUG_MEMORY:-1}"
GPU_TELEMETRY_ENABLE="${GPU_TELEMETRY_ENABLE:-1}"
GPU_TELEMETRY_INTERVAL_MS="${GPU_TELEMETRY_INTERVAL_MS:-200}"

CHECKPOINT="$(realpath -m "${CHECKPOINT}")"
LOGDIR="$(realpath -m "${LOGDIR}")"

if [[ ! -f "${CHECKPOINT}/config.json" ]]; then
  echo "[smoke-long] ERROR: ${CHECKPOINT}/config.json not found"
  exit 1
fi
if ! compgen -G "${CHECKPOINT}/model*.pth" >/dev/null; then
  echo "[smoke-long] ERROR: no model*.pth found in ${CHECKPOINT}"
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
ET.ElementTree(root).write(out_file, encoding="utf-8", xml_declaration=True)
print(f"[smoke-long] Built merged route file: {out_file} ({route_id} routes)")
PY
}

if [[ "${ROUTE_PROFILE}" == "town03_debug" ]]; then
  ROUTE_FILE="${ROUTE_FILE:-${CARL_ROOT}/custom_leaderboard/leaderboard/data/debug_routes_with_scenarios/route_Town03_00.xml.gz}"
elif [[ "${ROUTE_PROFILE}" == "debug_suite" ]]; then
  ROUTE_FILE="${ROUTE_FILE:-/tmp/tfv6_debug_suite_routes.xml}"
  build_debug_suite_routes "${ROUTE_FILE}"
elif [[ "${ROUTE_PROFILE}" == "training" ]]; then
  ROUTE_FILE="${ROUTE_FILE:-${CARL_ROOT}/custom_leaderboard/leaderboard/data/routes_training.xml}"
  # routes_training already has many routes; repetitions can stay low.
  REPETITIONS="${REPETITIONS:-1}"
else
  echo "[smoke-long] ERROR: unsupported ROUTE_PROFILE='${ROUTE_PROFILE}'"
  echo "Supported: town03_debug | debug_suite | training"
  exit 1
fi

ROUTE_FILE="$(realpath -m "${ROUTE_FILE}")"
if [[ ! -f "${ROUTE_FILE}" ]]; then
  echo "[smoke-long] ERROR: route file not found: ${ROUTE_FILE}"
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

echo "[smoke-long] checkpoint=${CHECKPOINT}"
echo "[smoke-long] route_profile=${ROUTE_PROFILE}"
echo "[smoke-long] route_file=${ROUTE_FILE}"
echo "[smoke-long] repetitions=${REPETITIONS}"
echo "[smoke-long] total_timesteps=${TOTAL_TIMESTEPS}"
echo "[smoke-long] total_batch_size=${TOTAL_BATCH_SIZE}"
echo "[smoke-long] total_minibatch_size=${TOTAL_MINIBATCH_SIZE}"
echo "[smoke-long] update_epochs=${UPDATE_EPOCHS}"
echo "[smoke-long] track=${TRACK}"
echo "[smoke-long] debug_memory=${DEBUG_MEMORY}"
echo "[smoke-long] logs_dir=${RUN_DIR}"

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
  echo "[smoke-long] gpu_telemetry=${GPU_TELEMETRY_LOG}"
  echo "[smoke-long] gpu_processes=${GPU_APPS_LOG}"
else
  echo "[smoke-long] GPU telemetry disabled (set GPU_TELEMETRY_ENABLE=1 and ensure nvidia-smi is available)."
fi

echo "[smoke-long] Starting leaderboard client (expects CARLA server on port ${CARLA_PORT})"
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
  2>&1 | awk '{ print strftime("[%Y-%m-%d %H:%M:%S]"), $0; fflush(); }' >"${LEADERBOARD_LOG}" &
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

echo "[smoke-long] Starting TFv6 PPO trainer"
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
  2>&1 | awk '{ print strftime("[%Y-%m-%d %H:%M:%S]"), $0; fflush(); }' | tee "${TRAINER_LOG}"

echo "[smoke-long] Done. Check TensorBoard logs at ${LOGDIR}/${EXP_NAME}"
echo "[smoke-long] Leaderboard log: ${LEADERBOARD_LOG}"
echo "[smoke-long] Trainer log: ${TRAINER_LOG}"
