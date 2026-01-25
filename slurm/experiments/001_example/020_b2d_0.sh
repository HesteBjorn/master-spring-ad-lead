#!/usr/bin/bash

export CARLA_ROOT=3rd_party/CARLA_0915
export PYTHONPATH="$CARLA_ROOT/PythonAPI:$CARLA_ROOT/PythonAPI/carla:$PWD:$PYTHONPATH"
# SLURM generator parameters for per-route jobs
# export EVALUATION_PARAMETERS="--id_list 10857"  # COMMENT OUT THIS FOR FULL RUN
export SCRIPT_GENERATOR_PARAMETERS="--partition=GPUQ --slurm_timeout 0-01:00:00 --repetitions 1"
export WANDB_MODE=disabled

source slurm/init.sh

export CHECKPOINT_DIR=outputs/checkpoints/tfv6_resnet34
export LEAD_TRAINING_CONFIG="$LEAD_TRAINING_CONFIG"
export LEAD_CLOSED_LOOP_CONFIG="$LEAD_CLOSED_LOOP_CONFIG"

evaluate_bench2drive
