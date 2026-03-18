#!/usr/bin/bash
#SBATCH --job-name=tfv6_ppo_parallel_smoke
#SBATCH --account=share-ie-idi
#SBATCH --mail-user=erikhbj@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-task=48
#SBATCH --constraint=a100|h100|h200
#SBATCH --mem=256G
#SBATCH --time=0-00:15:00
#SBATCH --output=outputs/rl_logs/slurm_%j.out
#SBATCH --error=outputs/rl_logs/slurm_%j.err

set -eo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export LEAD_PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"

# Keep behavior consistent with existing experiment scripts.
source slurm/init.sh

export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-disabled}"
export CARLA_ROOT="${CARLA_ROOT:-$PWD/3rd_party/CARLA_0915}"
export PYTHONPATH="$CARLA_ROOT/PythonAPI:$CARLA_ROOT/PythonAPI/carla:$PWD:$PYTHONPATH"

# Module/conda setup (IDUN-specific)
module purge
module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
source activate "${CONDA_INTERPRETER:-lead}"

mkdir -p outputs/rl_logs

echo "[slurm] Job ID: ${SLURM_JOB_ID:-local}"
nvidia-smi

GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
GPU_IDS="$(seq -s ' ' 0 $((GPU_COUNT - 1)))"
GPU_MEM_MIB="$(nvidia-smi --id=0 --query-gpu=memory.total --format=csv,noheader,nounits)"

# Auto-pick envs/GPU from detected VRAM unless user overrides NUM_ENVS_PER_GPU.
# Targets:
# - 80GB class: 4 envs/GPU
# - 40GB class: 3 envs/GPU
# - 24-39GB class: 2 envs/GPU
# - smaller cards: 1 env/GPU
if [[ "${NUM_ENVS_PER_GPU:-}" == "" ]]; then
  if (( GPU_MEM_MIB >= 76000 )); then
    NUM_ENVS_PER_GPU=4
  elif (( GPU_MEM_MIB >= 38000 )); then
    NUM_ENVS_PER_GPU=3
  elif (( GPU_MEM_MIB >= 24000 )); then
    NUM_ENVS_PER_GPU=2
  else
    NUM_ENVS_PER_GPU=1
  fi
fi
# TEMPORARY OVERRIDE TO TEST:
NUM_ENVS_PER_GPU=1

NUM_ENVS_PER_NODE="${NUM_ENVS_PER_NODE:-$((GPU_COUNT * NUM_ENVS_PER_GPU))}"

RUN_CONFIG_FILE="${RUN_CONFIG_FILE:-rl_finetuning/configs/train_parallel_idun.env}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-outputs/checkpoints/tfv6_resnet34}"
LOGDIR="${LOGDIR:-outputs/rl_logs}"
EXP_NAME="${EXP_NAME:-TFV6_PPO_PARALLEL_SLURM_${SLURM_JOB_ID:-manual}}"

echo "[slurm] gpu_count=${GPU_COUNT} gpu_ids=${GPU_IDS}"
echo "[slurm] gpu_mem_mib=${GPU_MEM_MIB} num_envs_per_gpu=${NUM_ENVS_PER_GPU} num_envs_per_node=${NUM_ENVS_PER_NODE}"
echo "[slurm] run_config=${RUN_CONFIG_FILE}"
echo "[slurm] checkpoint=${CHECKPOINT_DIR} logdir=${LOGDIR} exp_name=${EXP_NAME}"

RUN_CONFIG_FILE="${RUN_CONFIG_FILE}" \
NUM_ENVS_PER_NODE="${NUM_ENVS_PER_NODE}" \
NUM_ENVS_PER_GPU="${NUM_ENVS_PER_GPU}" \
GPU_IDS="${GPU_IDS}" \
bash rl_finetuning/train_tfv6_ppo_parallell.sh \
  "${CHECKPOINT_DIR}" \
  "${LOGDIR}" \
  "${EXP_NAME}" \
  --debug 1
