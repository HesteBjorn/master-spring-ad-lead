#!/bin/bash
set -euo pipefail

## RUN: Specify runs in the bottom part of the file to execute.

# Make CARLA's PythonAPI importable (provides the `agents.navigation.*` package
# that lead.common.common_utils and the leaderboard depend on). Without this,
# extraction and eval fail with `ModuleNotFoundError: No module named 'agents'`.
# Override CARLA_ROOT before calling if you use a different CARLA tree.
: "${CARLA_ROOT:=3rd_party/CARLA_0915}"
export PYTHONPATH="$CARLA_ROOT/PythonAPI/carla:$CARLA_ROOT/PythonAPI:${PYTHONPATH:-}"

# Returns (via echo) the checkpoint directory ready for eval.
# If given an outputs/checkpoints/* dir, returns it as-is.
# If given an outputs/rl_logs/*/<model>.pth, derives outputs/checkpoints/<run>_<step>
# and runs extraction if that dir doesn't exist yet.
resolve_checkpoint_dir() {
    local checkpoint="$1"

    # Direct checkpoint dir — return as-is
    if [[ "$checkpoint" != *.pth ]]; then
        echo "$checkpoint"
        return
    fi

    # rl_logs path: derive name from <run_folder>_<step> e.g. TD3_idun_baseline_000001500000
    local model_file run_dir step_str out_dir
    model_file="$(basename "$checkpoint" .pth)"          # model_latest_000001500000
    step_str="${model_file##*_}"                         # 000001500000
    run_dir="$(basename "$(dirname "$checkpoint")")"     # TD3_idun_baseline
    out_dir="outputs/checkpoints/${run_dir}_${step_str}"

    # Check for a valid prior extraction (not just the dir, which may be a partial/failed run)
    if [[ -f "$out_dir/residual_policy.pth" || -f "$out_dir/td3_policy.pth" ]]; then
        echo "$out_dir"
        return
    fi

    # Read tfv6_checkpoint from the run config and strip the machine-specific prefix,
    # keeping only the outputs/checkpoints/... suffix for local resolution.
    local run_config base_checkpoint_dir base_arg=()
    run_config="$(dirname "$checkpoint")/config.json"
    if [[ -f "$run_config" ]]; then
        local stored_base
        stored_base="$(python3 -c "import json,sys; print(json.load(open('$run_config')).get('tfv6_checkpoint',''))")"
        if [[ -n "$stored_base" ]]; then
            base_checkpoint_dir="${stored_base#*outputs/checkpoints/}"
            base_checkpoint_dir="outputs/checkpoints/${base_checkpoint_dir}"
            base_arg=(--base-checkpoint-dir "$base_checkpoint_dir")
            echo "[resolve_checkpoint] Using base checkpoint: $base_checkpoint_dir" >&2
        fi
    fi

    echo "[resolve_checkpoint] Extracting $checkpoint -> $out_dir" >&2
    # Redirect stdout to stderr so diagnostic prints from the script don't pollute $()
    python3 scripts/extract_trained_tfv6_model_from_policy.py \
        "$checkpoint" \
        --output-dir "$out_dir" \
        "${base_arg[@]}" >&2

    echo "$out_dir"
}

# Function to run the run.
# $1 may be a single checkpoint, OR a comma-separated list of TD3 residual
# checkpoints. With more than one, they are auto-ensembled (coefficient mean,
# shared frozen TFv6) via the td3_ensemble_sensor_agent — the $2 agent arg is
# overridden and a clean output-dir name is used. Single-model runs are unchanged.
run_b2d_with_chp_agent_route() {
    local agent="$2" routes="$3"

    # Split $1 on commas into the list of requested checkpoints, resolve each.
    local _ckpts resolved=() c
    IFS=',' read -ra _ckpts <<< "$1"
    for c in "${_ckpts[@]}"; do
        [[ -n "$c" ]] && resolved+=("$(resolve_checkpoint_dir "$c")")
    done

    local checkpoint eval_name route_id
    route_id="$(basename "$routes" .xml)"
    if [[ "${#resolved[@]}" -gt 1 ]]; then
        checkpoint="$(IFS=','; echo "${resolved[*]}")"  # dirA,dirB,dirC
        agent="rl_finetuning/inference/td3_ensemble_sensor_agent.py"
        eval_name="ensemble_${#resolved[@]}__$(basename "${resolved[0]}")"
        echo ""
        echo "=== ENSEMBLE       : ${#resolved[@]} models"
    else
        checkpoint="${resolved[0]}"
        eval_name="$(basename "$checkpoint")"
        echo ""
    fi
    echo "=== CHECKPOINT_DIR : $checkpoint"
    echo "=== AGENT          : $agent"
    echo "=== ROUTES         : $routes"

    trap 'bash scripts/clean_carla.sh' EXIT

    # Launch run
    bash scripts/clean_carla.sh  # ensure no stale CARLA/TM processes from a prior run
    bash scripts/start_carla.sh
    sleep 10
    CHECKPOINT_DIR="$checkpoint" AGENT="$agent" ROUTES="$routes" \
        EVALUATION_OUTPUT_DIR="outputs/local_evaluation/$route_id/$eval_name" \
        bash scripts/eval_bench2drive.sh
    bash scripts/clean_carla.sh
    sleep 5

    trap - EXIT
}


## RUNS:
# Add/comment runs below — one call = one evaluation

run_b2d_with_chp_agent_route \
    "outputs/rl_logs/TD3_idun_baseline_allimprovements/model_latest_000001500000.pth" \
    "rl_finetuning/inference/td3_sensor_agent.py" \
    "data/benchmark_routes/bench2drive20LT.xml"

run_b2d_with_chp_agent_route \
    "outputs/rl_logs/TFV6_TD3_LOCAL_RESIDUAL_improved_T1213_randomweather/model_latest_000001500000.pth" \
    "rl_finetuning/inference/td3_sensor_agent.py" \
    "data/benchmark_routes/bench2drive20LT.xml"

run_b2d_with_chp_agent_route \
    "outputs/rl_logs/TFV6_TD3_LOCAL_RESIDUAL_improved_T1213_all4lt/model_latest_000002000000.pth" \
    "rl_finetuning/inference/td3_sensor_agent.py" \
    "data/benchmark_routes/bench2drive20LT.xml"


 # Ensemble: pass a comma-separated list as $1 (no spaces).
# # The models are blended by averaging residual coefficients over a shared frozen TFv6 base.
# # The agent arg is ignored here (auto-set to td3_ensemble_sensor_agent.py).
# Ensamble models using "," between checkpoints.
# run_b2d_with_chp_agent_route \
#     "outputs/rl_logs/TD3_idun_baseline_allimprovements/model_latest_000001500000.pth," \
#     "rl_finetuning/inference/td3_sensor_agent.py" \
#     "data/benchmark_routes/bench2drive20LT.xml"

# # For PPO:
# "rl_finetuning/inference/residual_sensor_agent.py"

# run_b2d_with_chp_agent_route \
#     "outputs/checkpoints/tfv6_resnet34" \
#     "lead/inference/sensor_agent.py" \
#     "data/benchmark_routes/bench2drive20LT.xml"

# run_b2d_with_chp_agent_route \
#     "outputs/checkpoints/tfv6_residual_onlyspeed_latest" \
#     "rl_finetuning/inference/residual_sensor_agent.py" \
#     "data/benchmark_routes/bench2drive20LT.xml"

bash scripts/clean_carla.sh
