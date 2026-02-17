#!/bin/bash

export repo_root=$1
export carl_root=$2
export route_file=$3
export logdir=$4
export index=$5
export client_port=$6
export tm_port=$7
export rl_port=$8
export random_seed=$9
export skip_next_route=${10}
export repetitions=${11}
export tfv6_checkpoint=${12}
export track=${13}
export frame_rate=${14}
export no_rendering_mode=${15}
export timeout=${16}
export runtime_timeout=${17}
export leaderboard_debug=${18}

export NUMEXPR_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export TFV6_CHECKPOINT=${tfv6_checkpoint}

python -u ${carl_root}/custom_leaderboard/leaderboard/leaderboard/leaderboard_evaluator.py --routes ${route_file} --agent ${repo_root}/rl_finetuning/tfv6_rl/env_agent_tfv6.py --checkpoint ${logdir}/route_${index}.json --track ${track} --port ${client_port} --traffic-manager-port ${tm_port} --agent-config ${tfv6_checkpoint} --gym_port ${rl_port} --traffic-manager-seed ${random_seed} --skip_next_route ${skip_next_route} --frame_rate ${frame_rate} --resume 1 --no_rendering_mode ${no_rendering_mode} --runtime_timeout ${runtime_timeout} --timeout ${timeout} --repetitions ${repetitions} --debug ${leaderboard_debug}
