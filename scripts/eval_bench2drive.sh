#!/bin/bash

# Checkpoints
# export CHECKPOINT_DIR=outputs/checkpoints/tfv6_resnet34/  # outputs/checkpoints/tfv6_resnet34_rlfinetuned_modelbest
# export AGENT=lead/inference/sensor_agent.py
export CHECKPOINT_DIR=outputs/checkpoints/tfv6_residual_onlyspeed_latest
export AGENT=rl_finetuning/inference/residual_sensor_agent.py  # NEED TO BE ACTIVE TO USE RESIDUAL

# export ROUTES=data/benchmark_routes/bench2drive10routesexampleonlyfails.xml
export ROUTES=data/benchmark_routes/bench2drive10NSLTEF.xml  # for left-turn eval
# export ROUTES=data/benchmark_routes/bench2drive5NSRT.xml  # For right-turn eval
# export ROUTES=data/benchmark_routes/bench2drive220routes/23687.xml  # For single route eval

# Set environment variables
export BENCHMARK_ROUTE_ID=$(basename $ROUTES .xml) # Last part of the route file name, e.g., 0 for 0.xml
export EVALUATION_OUTPUT_DIR=outputs/local_evaluation/$BENCHMARK_ROUTE_ID/$(basename $CHECKPOINT_DIR)
export PYTHONPATH=3rd_party/Bench2Drive/leaderboard:$PYTHONPATH
export PYTHONPATH=3rd_party/Bench2Drive/scenario_runner:$PYTHONPATH
export SCENARIO_RUNNER_ROOT=3rd_party/Bench2Drive/scenario_runner
export LEADERBOARD_ROOT=3rd_party/Bench2Drive/leaderboard
export IS_BENCH2DRIVE=1
export PLANNER_TYPE=only_traj
export SAVE_PATH=$EVALUATION_OUTPUT_DIR/
export PYTHONUNBUFFERED=1

set -x
set +e

# Recreate output folders
rm -rf $EVALUATION_OUTPUT_DIR/
mkdir -p $EVALUATION_OUTPUT_DIR

CUDA_VISIBLE_DEVICES=0 python3 3rd_party/Bench2Drive/leaderboard/leaderboard/leaderboard_evaluator.py \
    --routes=$ROUTES \
    --track=SENSORS \
    --checkpoint=$EVALUATION_OUTPUT_DIR/checkpoint_endpoint.json \
    --agent=$AGENT \
    --agent-config=$CHECKPOINT_DIR \
    --debug=0 \
    --record=None \
    --resume=False \
    --port=2000 \
    --traffic-manager-port=8000 \
    --timeout=60 \
    --debug-checkpoint=$EVALUATION_OUTPUT_DIR/debug_checkpoint/debug_checkpoint_endpoint.txt \
    --traffic-manager-seed=0 \
    --repetitions=3
