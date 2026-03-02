# Evaluation

## Extract policy checkpoint to eval-compatible TFv6 checkpoint
```bash
python3 scripts/extract_trained_tfv6_model_from_policy.py \
  outputs/rl_logs/TFV6_PPO_PARALLEL_LOCAL_MULTIENV_SINGLEGPU_33_NEWSTDHEAD/model_best.pth \
  --output-dir outputs/checkpoints/tfv6_resnet34_rlfinetuned_modelbest
```

## Bench2Drive Idun
From repo root in idun run: `bash slurm/experiments/001_example/020_b2d_0.sh` to create many slurm jobs for B2D.

`cd /outputs/evaluation/001_example/020_b2d_0/` for å finne resultater fra idun.

## Run b2d dashboard from idun files
```bash
# Login to idun and port forward
ssh -L 5000:localhost:5000 erikhbj@idun.hpc.ntnu.no
module purge
module load Anaconda3/2024.02-1
conda activate lead
```
```bash
# On idun
cd /cluster/home/erikhbj/master/master-spring-ad-lead
module purge
module load Anaconda3/2024.02-1
conda activate lead
# Auto update video folder reference to newest folder
cd /cluster/home/erikhbj/master/master-spring-ad-lead/outputs && rm -rf local_evaluation && ln -s "$(ls -td evaluation/001_example/020_b2d_0/* | head -n 1)" local_evaluation && ls -l local_evaluation
# Start webapp
cd /cluster/home/erikhbj/master/master-spring-ad-lead
python lead/infraction_webapp/app.py
```

## Local debug route
```bash
# Load conda
conda activate lead
# Clean old carla
bash scripts/clean_carla.sh
# Start driving environment
bash scripts/start_carla.sh
# Start policy on one route
bash scripts/eval_bench2drive.sh
```

# Training
## Generate scenarios:
```bash
#Use the CaRL generator:
cd 3rd_party/CaRL/CARLA/tools && \
  python -u generate_long_routes_with_scenarios.py \
    --save_folder /home/erikhbj/Documents/master/master-spring-ad-lead/3rd_party/CaRL/CARLA/custom_leaderboard/leaderboard/data/rl_finetuning_1000_meters_alltypes_dense80 \
    --carla_root /home/erikhbj/Documents/master/master-spring-ad-lead/3rd_party/CARLA_0915 \
    --scenario_runner_root /home/erikhbj/Documents/master/master-spring-ad-lead/3rd_party/CaRL/CARLA/custom_leaderboard/scenario_runner \
    --start_repetition 0 \
    --scenario_dilation 80 \
    --generate_scenarios 1 \
    --only_leaderboard_1 0 \
    --route_length 1000
```

## Visualize losses and rewards
```bash
tensorboard --logdir outputs/rl_logs
```

## Parallell training
### Local parallell training
```bash
# Multi env in parallel
RUN_CONFIG_FILE=rl_finetuning/configs/train_parallel_singleGPU.env \
bash rl_finetuning/train_tfv6_ppo_parallell.sh \
  outputs/checkpoints/tfv6_resnet34 \
  outputs/rl_logs \
  TFV6_PPO_PARALLEL_LOCAL_MULTIENV_SINGLEGPU \
  --debug 1
```

### Slurm parallell training
```bash
sbatch --export=ALL,RUN_CONFIG_FILE=rl_finetuning/configs/train_parallel_idun.env rl_finetuning/train_tfv6_ppo_parallell_slurm.slurm
```
```bash
# Idun check queue status
scontrol show job -dd 24060485 | egrep "JobId=|Partition=|QOS=|A
ccount=|UserId=|Priority=|Reason=|StartTime=|EligibleTime=|ReqTRES=|NumNodes=|NumCPUs=|Mi
nMemory="

# Check status for all in queue
squeue -u erikhbj -h -t PD -o "%i" | xargs -r -n1 -I{} sh -c 'echo "===== JOB {} =====";scontrol show job -dd {} | egrep "JobId=|Partition=|QOS=|Account=|UserId=|Priority=|Reason=|StartTime=|EligibleTime=|ReqTRES=| NumNodes=|NumCPUs=|MinMemory="'
```

### Kill all running processes
```bash
pkill -f "train_tfv6_ppo_parallell.sh" || true
pkill -f "train_parallel_tfv6_ppo.py" || true
pkill -f "start_learner_dd_ppo_tfv6_ppo.sh" || true
pkill -f "torchrun.*rl_finetuning/train_tfv6_ppo.py" || true
pkill -f "leaderboard_evaluator.py" || true
pkill -f "start_leaderboard_tfv6_ppo.sh" || true
pkill -f "CarlaUE4-Linux-Shipping" || true
pkill -f "CarlaUE4.sh" || true
ps -ef | rg "CarlaUE4|leaderboard_evaluator|train_parallel_tfv6_ppo|train_tfv6_ppo.py|torchrun|tensorboard" || true
```

# Analysis

## Debug viz folder to mp4
```bash
rl_finetuning/m4p_convert_folder.sh outputs/rl_logs/TFV6_PPO_WEEKEND_20260213/run_20260213_144404/debug_viz
```

## Analyse action distribution shift from TFv6 to Fine tuned model:
Update data-root with the most recent checkpoint
```bash
python rl_finetuning/analyze_speed_distribution_shift.py \
--old-checkpoint outputs/checkpoints/tfv6_resnet34/model_0030_0.pth \
--finetuned-checkpoint outputs/rl_logs/TFV6_PPO_WEEKEND_20260214_correlationrho095_logstd45/model_latest_000000099.pth \
--data-root data/carla_leaderboard2/data \
--output-dir outputs/local_evaluation \
--output-stem speed_distribution_shift_all_routes
```

# Div
## SSH with vscode to 5090 computer
Connect to eduroam network (or VPN)
```bash
# Tailscale IP: erikhbj@100.92.150.98
# Trenger ikke være på VPN. Anbefalt ikke.
ssh erikhbj@100.92.150.98
```
Or go to vscode: `Ctrl+Shift+P> Remote-SSH: Connect to Host...`

## Git push logs from idun excluding .pth files
```bash
# From root, specify rundir
RUN_DIR=outputs/rl_logs/TFV6_PPO_PARALLEL_SLURM_24056917
git add -f "$RUN_DIR" \
  ":(exclude)$RUN_DIR/model*.pth" \
  ":(exclude)$RUN_DIR/optimizer*.pth"


# Or: From rundir:
git add -f . \
  ':(exclude)model*.pth' \
  ':(exclude)optimizer*.pth'

# Confirm no model.pth are staged:
git status --short
```

## Tmux: Keep terminal alive across SSH
```bash
# Create new terminal
tmux new -s rltrain
# ... use terminal

# detach: Ctrl+b, then d
tmux ls
tmux attach -t rltrain
```
