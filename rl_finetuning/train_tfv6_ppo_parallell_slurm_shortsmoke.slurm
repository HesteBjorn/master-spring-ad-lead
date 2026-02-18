#!/usr/bin/bash
#SBATCH --job-name=tfv6_ppo_parallel_smoke
#SBATCH --account=share-ie-idi
#SBATCH --mail-user=erikhbj@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=0-00:20:00
#SBATCH --constraint=gpu16g|gpu32g|gpu40g|gpu80g
#SBATCH --output=outputs/rl_logs/slurm_%j.out
#SBATCH --error=outputs/rl_logs/slurm_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

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
GPU_MEM_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)"

# Target 2 envs/GPU when VRAM is sufficient. Fallback to 1 for small GPUs.
if [[ "${NUM_ENVS_PER_GPU:-}" == "" ]]; then
  if (( GPU_MEM_MIB < 24000 )); then
    NUM_ENVS_PER_GPU=1
  else
    NUM_ENVS_PER_GPU=2
  fi
fi

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
