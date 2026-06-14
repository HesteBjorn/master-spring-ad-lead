#!/bin/bash

# Defaults — override by exporting before calling, or edit here for standalone use
: "${CHECKPOINT_DIR:=outputs/checkpoints/tfv6_resnet34/}"  # outputs/checkpoints/tfv6_resnet34_rlfinetuned_modelbest
: "${AGENT:=lead/inference/sensor_agent.py}"
# AGENT=rl_finetuning/inference/residual_sensor_agent.py  # NEED TO BE ACTIVE TO USE RESIDUAL

: "${ROUTES:=data/benchmark_routes/bench2drive20LT.xml}"  # All left-turn for eval
# ROUTES=data/benchmark_routes/bench2drive10routesexampleonlyfails.xml
# ROUTES=data/benchmark_routes/bench2drive10NSLTEF.xml  # for left-turn eval
# ROUTES=data/benchmark_routes/bench2drive5NSRT.xml  # For right-turn eval
# ROUTES=data/benchmark_routes/bench2drive220routes/23687.xml  # For single route eval

# Set environment variables
export BENCHMARK_ROUTE_ID=$(basename $ROUTES .xml) # Last part of the route file name, e.g., 0 for 0.xml
# Default-assign so callers (e.g. the ensemble path) can pre-set a clean name;
# a comma-separated CHECKPOINT_DIR would otherwise produce a misleading basename.
: "${EVALUATION_OUTPUT_DIR:=outputs/local_evaluation/$BENCHMARK_ROUTE_ID/$(basename $CHECKPOINT_DIR)}"
export EVALUATION_OUTPUT_DIR
export PYTHONPATH=3rd_party/Bench2Drive/leaderboard:$PYTHONPATH
export PYTHONPATH=3rd_party/Bench2Drive/scenario_runner:$PYTHONPATH
# CARLA PythonAPI provides the `agents.navigation.*` package. Harmless if the
# parent already added it. Override CARLA_ROOT to use a different CARLA tree.
: "${CARLA_ROOT:=3rd_party/CARLA_0915}"
export PYTHONPATH=$CARLA_ROOT/PythonAPI/carla:$CARLA_ROOT/PythonAPI:$PYTHONPATH
export SCENARIO_RUNNER_ROOT=3rd_party/Bench2Drive/scenario_runner
export LEADERBOARD_ROOT=3rd_party/Bench2Drive/leaderboard
export IS_BENCH2DRIVE=1
export PLANNER_TYPE=only_traj
# NO_SAVE=1 disables all debug image/video output (used for fast outcome-only scans).
# Leaving SAVE_PATH unset makes config_closed_loop.save_path return None.
if [[ "${NO_SAVE:-0}" == "1" ]]; then
    unset SAVE_PATH
else
    export SAVE_PATH=$EVALUATION_OUTPUT_DIR/
fi
export PYTHONUNBUFFERED=1

set -x
set +e

# Recreate output folders. RESUME=1 (default) keeps the existing checkpoint so the
# leaderboard skips already-completed routes instead of starting over. Pass
# RESUME=0 to force a clean re-evaluation that wipes prior results first.
if [[ "${RESUME:-1}" == "1" ]]; then
    mkdir -p $EVALUATION_OUTPUT_DIR
    RESUME_FLAG=1
else
    rm -rf $EVALUATION_OUTPUT_DIR/
    mkdir -p $EVALUATION_OUTPUT_DIR
    RESUME_FLAG=False
fi

CUDA_VISIBLE_DEVICES=0 python3 3rd_party/Bench2Drive/leaderboard/leaderboard/leaderboard_evaluator.py \
    --routes=$ROUTES \
    --track=SENSORS \
    --checkpoint=$EVALUATION_OUTPUT_DIR/checkpoint_endpoint.json \
    --agent=$AGENT \
    --agent-config=$CHECKPOINT_DIR \
    --debug=0 \
    --record=None \
    --resume=$RESUME_FLAG \
    --port=2000 \
    --traffic-manager-port=8000 \
    --timeout=60 \
    --debug-checkpoint=$EVALUATION_OUTPUT_DIR/debug_checkpoint/debug_checkpoint_endpoint.txt \
    --traffic-manager-seed=0 \
    --repetitions=${REPETITIONS:-3}
