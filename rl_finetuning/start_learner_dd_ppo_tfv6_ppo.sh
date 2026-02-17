#!/bin/bash

export repo_root=$1
export num_processes=$2
export num_nodes=$3
export rdzv_addr=$4
export rdzv_port=$5

export NUMEXPR_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=8
export MASTER_ADDR=${rdzv_addr}

torchrun --start-method spawn --nproc_per_node=${num_processes} --nnodes=${num_nodes} --max_restarts=0 --rdzv-backend=c10d --rdzv-endpoint=${rdzv_addr}:${rdzv_port} ${repo_root}/rl_finetuning/train_tfv6_ppo.py "${@:6}"
