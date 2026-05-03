#!/usr/bin/bash
#SBATCH --job-name=tfv6_td3_baseline
#SBATCH --account=share-ie-idi
#SBATCH --mail-user=erikhbj@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=48
#SBATCH --constraint=a100|h100|h200
#SBATCH --mem=160G
#SBATCH --time=3-00:00:00
#SBATCH --output=outputs/rl_logs/slurm_%j.out
#SBATCH --error=outputs/rl_logs/slurm_%j.err

set -eo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export LEAD_PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"

# Keep behavior consistent with existing experiment scripts.
source slurm/init.sh

export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-disabled}"
export CARLA_ROOT="${CARLA_ROOT:-$PWD/3rd_party/fork_export_t1213_fixed/LinuxNoEditor}"
export PYTHONPATH="$CARLA_ROOT/PythonAPI:$CARLA_ROOT/PythonAPI/carla:$PWD:${PYTHONPATH:-}"

# Module/conda setup (IDUN-specific). Training must use the fork-compatible env.
module purge
module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
source activate "${CONDA_INTERPRETER:-lead_carla_fork}"

mkdir -p outputs/rl_logs

echo "[slurm-td3] Job ID: ${SLURM_JOB_ID:-local}"
nvidia-smi

GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
GPU_IDS="${GPU_IDS:-0}"
NUM_ENVS_PER_GPU="${NUM_ENVS_PER_GPU:-4}"
NUM_ENVS_PER_NODE="${NUM_ENVS_PER_NODE:-4}"

RUN_CONFIG_FILE="${RUN_CONFIG_FILE:-rl_finetuning/configs/train_idun_residual_td3_BASELINE.env}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-outputs/checkpoints/tfv6_resnet34}"
LOGDIR="${LOGDIR:-outputs/rl_logs}"
EXP_NAME="${EXP_NAME:-TFV6_TD3_IDUN_RESIDUAL_BASELINE_${SLURM_JOB_ID:-manual}}"

echo "[slurm-td3] gpu_count=${GPU_COUNT} gpu_ids=${GPU_IDS}"
echo "[slurm-td3] num_envs_per_gpu=${NUM_ENVS_PER_GPU} num_envs_per_node=${NUM_ENVS_PER_NODE}"
echo "[slurm-td3] run_config=${RUN_CONFIG_FILE}"
echo "[slurm-td3] checkpoint=${CHECKPOINT_DIR} logdir=${LOGDIR} exp_name=${EXP_NAME}"

RUN_CONFIG_FILE="${RUN_CONFIG_FILE}" \
NUM_ENVS_PER_NODE="${NUM_ENVS_PER_NODE}" \
NUM_ENVS_PER_GPU="${NUM_ENVS_PER_GPU}" \
GPU_IDS="${GPU_IDS}" \
bash rl_finetuning/train_tfv6_td3_parallel.sh \
  "${CHECKPOINT_DIR}" \
  "${LOGDIR}" \
  "${EXP_NAME}" \
  --debug 0
