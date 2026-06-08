#!/usr/bin/env bash
# Build the all_4_leftturn_Scenarios training route bundle (Town12 + Town13).
#
# Launches a dedicated CARLA server, runs the route converter via
#   generate_all_intersection_routes.sh --source-dataset all4lt
# then shuts the server back down.
#
# Output: data/rl_finetuning_training_data/all_4_leftturn_Scenarios
#         (route_Town12_00/01.xml.gz, route_Town13_00/01.xml.gz, manifest.*)
#
# Run from the repo root:
#   bash rl_finetuning/make_all_4_leftturn_routes.sh
#
# Override any of the variables below inline, e.g.:
#   CARLA_PORT=2010 TRAIN_TOWNS="12 13" bash rl_finetuning/make_all_4_leftturn_routes.sh

set -euo pipefail

REPO_ROOT="$(pwd)"
if [[ ! -f "rl_finetuning/generate_all_intersection_routes.sh" ]]; then
    echo "ERROR: run this from the repo root (rl_finetuning/ not found here)." >&2
    exit 1
fi

# ---- Config (edit if your paths/ports/env differ) -------------------------
# Conda env whose carla python matches the server build.
CONDA_ENV="${CONDA_ENV:-lead_carla_fork}"
# RPC port for the throwaway conversion server (must be free).
CARLA_PORT="${CARLA_PORT:-2000}"
# Server binary that can load Town12/Town13 (your training fork).
CARLA_SERVER="${CARLA_SERVER:-${REPO_ROOT}/3rd_party/fork_export_t1213_fixed/LinuxNoEditor/CarlaUE4.sh}"
# CARLA_ROOT only needs to provide PythonAPI/agents for the converter import.
export CARLA_ROOT="${CARLA_ROOT:-${REPO_ROOT}/3rd_party/CARLA_0915}"
# GPU index for the server.
GPU="${GPU:-0}"
# Towns to convert (fork supports 12/13; not 15).
TRAIN_TOWNS="${TRAIN_TOWNS:-12 13}"
# Seconds to wait for the server RPC port before giving up.
CARLA_BOOT_TIMEOUT="${CARLA_BOOT_TIMEOUT:-120}"
# ---------------------------------------------------------------------------

unset PYTHONPATH || true

if [[ ! -x "${CARLA_SERVER}" && ! -f "${CARLA_SERVER}" ]]; then
    echo "ERROR: CARLA server launcher not found: ${CARLA_SERVER}" >&2
    exit 1
fi
if [[ ! -d "${CARLA_ROOT}/PythonAPI/carla/agents" ]]; then
    echo "ERROR: CARLA_ROOT has no PythonAPI/carla/agents: ${CARLA_ROOT}" >&2
    exit 1
fi

echo "[make-all4lt] launching CARLA: ${CARLA_SERVER} (rpc ${CARLA_PORT}, gpu ${GPU})"
env -u LD_LIBRARY_PATH bash "${CARLA_SERVER}" \
    -carla-rpc-port="${CARLA_PORT}" \
    -nosound -RenderOffScreen -graphicsadapter="${GPU}" \
    -RPCThreads=8 -StreamingThreads=8 -SecondaryThreads=8 &
CARLA_PID=$!

cleanup() {
    echo "[make-all4lt] stopping CARLA (pid ${CARLA_PID})"
    kill "${CARLA_PID}" 2>/dev/null || true
    wait "${CARLA_PID}" 2>/dev/null || true
}
trap cleanup EXIT

echo "[make-all4lt] waiting up to ${CARLA_BOOT_TIMEOUT}s for the RPC port..."
for ((i = 1; i <= CARLA_BOOT_TIMEOUT; i++)); do
    if ! kill -0 "${CARLA_PID}" 2>/dev/null; then
        echo "ERROR: CARLA process exited during startup." >&2
        exit 1
    fi
    if (exec 3<>"/dev/tcp/127.0.0.1/${CARLA_PORT}") 2>/dev/null; then
        exec 3>&- 3<&-
        echo "[make-all4lt] RPC port open after ${i}s"
        break
    fi
    sleep 1
done
# Give the RPC server a moment past first accept before load_world calls.
sleep 5

echo "[make-all4lt] activating conda env '${CONDA_ENV}' and running conversion (towns: ${TRAIN_TOWNS})"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

CARLA_PORT="${CARLA_PORT}" CARLA_HOST="127.0.0.1" \
    bash rl_finetuning/generate_all_intersection_routes.sh \
        --source-dataset all4lt \
        --interleaved-2env \
        --train-towns "${TRAIN_TOWNS}"

echo "[make-all4lt] done -> data/rl_finetuning_training_data/all_4_leftturn_Scenarios"
