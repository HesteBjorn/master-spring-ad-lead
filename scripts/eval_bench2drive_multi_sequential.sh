#!/bin/bash
set -euo pipefail

## RUN: Specify runs in the bottom part of the file to execute.

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

# Function to run the run
run_b2d_with_chp_agent_route() {
    local agent="$2" routes="$3"
    local checkpoint
    checkpoint="$(resolve_checkpoint_dir "$1")"

    echo ""
    echo "=== CHECKPOINT_DIR : $checkpoint"
    echo "=== AGENT          : $agent"
    echo "=== ROUTES         : $routes"

    trap 'bash scripts/clean_carla.sh' EXIT

    # Launch run
    bash scripts/clean_carla.sh  # ensure no stale CARLA/TM processes from a prior run
    bash scripts/start_carla.sh
    sleep 10
    CHECKPOINT_DIR="$checkpoint" AGENT="$agent" ROUTES="$routes" bash scripts/eval_bench2drive.sh
    bash scripts/clean_carla.sh
    sleep 5

    trap - EXIT
}


## RUNS:

# Add/comment runs below — one call = one evaluation
run_b2d_with_chp_agent_route \
    "outputs/rl_logs/TD3_idun_baseline/model_latest_000001500000.pth" \
    "rl_finetuning/inference/td3_sensor_agent.py" \
    "data/benchmark_routes/bench2drive20LT.xml"

# # For PPO:
# "rl_finetuning/inference/residual_sensor_agent.py"

run_b2d_with_chp_agent_route \
    "outputs/checkpoints/tfv6_resnet34" \
    "lead/inference/sensor_agent.py" \
    "data/benchmark_routes/bench2drive20LT.xml"

# run_b2d_with_chp_agent_route \
#     "outputs/checkpoints/tfv6_residual_onlyspeed_latest" \
#     "rl_finetuning/inference/residual_sensor_agent.py" \
#     "data/benchmark_routes/bench2drive20LT.xml"

bash scripts/clean_carla.sh
