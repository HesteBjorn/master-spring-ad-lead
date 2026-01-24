#!/usr/bin/bash

export CARLA_ROOT=/cluster/projects/vc/data/ad/open/write-folder/carla_0.9.15
export PYTHONPATH="$CARLA_ROOT/PythonAPI:$CARLA_ROOT/PythonAPI/carla:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg:$PYTHONPATH"
export PYTHONPATH="$PWD:$PYTHONPATH"
# SLURM generator parameters for per-route jobs
export SCRIPT_GENERATOR_PARAMETERS="--partition=GPUQ --slurm_timeout 0-01:00:00 --repetitions 1"

source slurm/init.sh

export CHECKPOINT_DIR=outputs/checkpoints/tfv6_resnet34
export LEAD_TRAINING_CONFIG="$LEAD_TRAINING_CONFIG"
export LEAD_CLOSED_LOOP_CONFIG="$LEAD_CLOSED_LOOP_CONFIG"

evaluate_bench2drive
