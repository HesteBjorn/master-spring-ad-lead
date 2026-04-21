#!/bin/bash
# Thin launcher for TFv6 TD3 single-process training.
# Called by train_tfv6_td3_parallel.sh (mirrors start_learner_dd_ppo_tfv6_ppo.sh).
# No torchrun — TD3 is single-process.

export repo_root=$1

export NUMEXPR_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=8

python -u ${repo_root}/rl_finetuning/train_tfv6_td3.py "${@:2}"
