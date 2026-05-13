# Evaluation

## Extract policy checkpoint to eval-compatible TFv6 checkpoint
```bash
python3 scripts/extract_trained_tfv6_model_from_policy.py \
  outputs/rl_logs/TFV6_LOCAL_RESIDUAL_onlyspeed/model_latest_000001821.pth \
  --output-dir outputs/checkpoints/tfv6_residual_onlyspeed_latest
```

## Local eval bench2drive
```bash
bash scripts/start_carla.sh ; sleep 10 ; export CHECKPOINT_DIR=outputs/checkpoints/tfv6_residual_onlyspeed_latest && AGENT=rl_finetuning/inference/residual_sensor_agent.py bash scripts/eval_bench2drive.sh ;  bash scripts/clean_carla.sh
```

## Bench2Drive Idun
From repo root in idun run: `bash slurm/experiments/001_example/020_b2d_0.sh` to create many slurm jobs for B2D.

`cd /outputs/evaluation/001_example/020_b2d_0/` for å finne resultater fra idun.

## Run b2d dashboard from idun files
```bash
# Login to idun and port forward
ssh -L 6007:localhost:6007 erikhbj@idun.hpc.ntnu.no
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
# Convert all intersection LEAD routes to RL compatible:
# conda activate lead
# export CARLA_ROOT=/home/erikhbj/Documents/master/master-spring-ad-lead/3rd_party/CARLA_0915
conda activate lead_carla_fork
export CARLA_ROOT=/home/erikhbj/Documents/master/master-spring-ad-lead/3rd_party/fork_export_t1213_fixed/LinuxNoEditor
unset PYTHONPATH
bash scripts/start_carla.sh &
sleep 15
bash rl_finetuning/generate_all_intersection_routes.sh --train-towns "12 13" --source-dataset 50x --interleaved-2env
# bash rl_finetuning/generate_all_intersection_routes.sh --train-towns "3 4 5 12 13" --source-dataset 50x --interleaved-2env  # For two envs on same town
bash scripts/clean_carla.sh
```

## Visualize losses and rewards
```bash
tensorboard --logdir outputs/rl_logs
```

## Parallell training
### PPO Local
```bash
# Multi env in parallel
conda activate lead
CARLA_ROOT=/home/erikhbj/Documents/master/master-spring-ad-lead/3rd_party/CARLA_0915 \
RUN_CONFIG_FILE=rl_finetuning/configs/train_parallel_singleGPU.env \
bash rl_finetuning/train_tfv6_ppo_parallell.sh \
  outputs/checkpoints/tfv6_resnet34 \
  outputs/rl_logs \
  TFV6_PPO_PARALLEL_LOCAL_newTP_debug3 \
  ---debug 0 --debug_viz --debug_viz_every_n 500000 --debug_viz_burst_len 2000
```
```bash
# Carla fork run
conda activate lead_carla_fork
export CARLA_ROOT=/home/erikhbj/Documents/master/master-spring-ad-lead/3rd_party/fork_export_t1213_fixed/LinuxNoEditor
# export CARLA_ROOT=/home/erikhbj/Documents/master/master-spring-ad-lead/3rd_party/fork_export/runtime/LinuxNoEditor   # First original working version before town12 and 13 fix.
# Avoid mixing in an old CARLA Python path from another shell session
unset PYTHONPATH
RUN_CONFIG_FILE=rl_finetuning/configs/train_parallel_singleGPU_singlescenario.env \
bash rl_finetuning/train_tfv6_ppo_parallell.sh \
  outputs/checkpoints/tfv6_resnet34 \
  outputs/rl_logs \
  TFV6_PPO_LOCAL_only_NSLTEF_warmstart_routedevpenal_terminalwarmupn5 \
  ---debug 0 --debug_viz --debug_viz_every_n 500000 --debug_viz_burst_len 2000
  # RUN_CONFIG_FILE=rl_finetuning/configs/train_parallel_singleGPU.env \
  # TFV6_PPO_PARALLEL_LOCAL_carlafork_newspeedsample_holdspeed_fixTPflip_t1_2 \
```
```bash
# RESIDUAL RUN
conda activate lead_carla_fork
export CARLA_ROOT=/home/erikhbj/Documents/master/master-spring-ad-lead/3rd_party/fork_export_t1213_fixed/LinuxNoEditor
unset PYTHONPATH
RUN_CONFIG_FILE=rl_finetuning/configs/train_local_residual.env  \
bash rl_finetuning/train_tfv6_ppo_parallell.sh \
  outputs/checkpoints/tfv6_resnet34 \
  outputs/rl_logs \
  TFV6_LOCAL_RESIDUAL_onlyspeed_newarchitecture_CNN \
  ---debug 0 --debug_viz --debug_viz_every_n 500000 --debug_viz_burst_len 2000
```
#### To continue the newarchitecture_CNN run:
For the outputs/rl_logs/TFV6_LOCAL_RESIDUAL_onlyspeed_newarchitecture_CNN run.
Temporarily change line 1292 in train_tfv6_ppo.py from strict=True to strict=False for this one resume, then change it back. Or add a dedicated --strict_load flag.
Since architecture have been slightly modified in the meantime (std head was changed to parameter)

```bash
# VC06 RESIDUAL
source ~/.bashrc
conda activate lead_carla_fork
export CARLA_ROOT=/data/work/erikhbj/master-spring-ad-lead/3rd_party/fork_export_t1213_fixed/LinuxNoEditor
unset PYTHONPATH
RUN_CONFIG_FILE=rl_finetuning/configs/train_vc06_ppo_residual.env  \
bash rl_finetuning/train_tfv6_ppo_parallell.sh \
  outputs/checkpoints/tfv6_resnet34 \
  outputs/rl_logs \
  TFV6_PPO_VC06_RESIDUAL_onlyspeed_CNN_fork \
  ---debug 0 --debug_viz --debug_viz_every_n 500000 --debug_viz_burst_len 2000

# VC06 nofork
source ~/.bashrc
conda activate lead
export CARLA_ROOT=/data/work/erikhbj/master-spring-ad-lead/3rd_party/CARLA_0915
unset PYTHONPATH
RUN_CONFIG_FILE=rl_finetuning/configs/train_vc06_ppo_residual.env  \
bash rl_finetuning/train_tfv6_ppo_parallell.sh \
  outputs/checkpoints/tfv6_resnet34 \
  outputs/rl_logs \
  TFV6_PPO_VC06_RESIDUAL_onlyspeed_CNN_nofork \
  --debug 0 --debug_viz --debug_viz_every_n 100000 --debug_viz_burst_len 2000
```

## TD3 Local
```bash
conda activate lead_carla_fork
export CARLA_ROOT=/home/erikhbj/Documents/master/master-spring-ad-lead/3rd_party/fork_export_t1213_fixed/LinuxNoEditor
unset PYTHONPATH
RUN_CONFIG_FILE=rl_finetuning/configs/train_local_residual_td3.env \
bash rl_finetuning/train_tfv6_td3_parallel.sh \
  outputs/checkpoints/tfv6_resnet34 \
  outputs/rl_logs \
  TFV6_TD3_LOCAL_RESIDUAL_1 \
  --debug 0 --debug_viz --debug_viz_every_n 20000 --debug_viz_burst_len 2000
```

## Kill all running processes
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

### Slurm parallell training
```bash
# Residual
sbatch --export=ALL,RUN_CONFIG_FILE=rl_finetuning/configs/train_parallel_residual_idun.env,EXP_NAME=TFV6_RESIDUAL_ONLYSPEED_IDUN_2  rl_finetuning/train_tfv6_ppo_parallell_slurm.slurm
```
```bash
# Finetuning
sbatch --export=ALL,RUN_CONFIG_FILE=rl_finetuning/configs/train_parallel_idun.env rl_finetuning/train_tfv6_ppo_parallell_slurm.slurm
```
```bash
# Computesmoke
sbatch --export=ALL,RUN_CONFIG_FILE=rl_finetuning/configs/train_parallel_idun.env rl_finetuning/train_tfv6_ppo_parallell_slurm_shortsmoke.slurm
```
```bash
# Idun check queue status
scontrol show job -dd 24060485 | egrep "JobId=|Partition=|QOS=|Account=|UserId=|Priority=|Reason=|StartTime=|EligibleTime=|ReqTRES=|NumNodes=|NumCPUs=|MinMemory="

# Check status for all in queue
squeue -u erikhbj -h -t PD -o "%i" | xargs -r -n1 -I{} sh -c 'echo "===== JOB {} =====";scontrol show job -dd {} | egrep "JobId=|Partition=|QOS=|Account=|UserId=|Priority=|Reason=|StartTime=|EligibleTime=|ReqTRES=| NumNodes=|NumCPUs=|MinMemory="'
```
#### Transfer slurm logs to local machine (run from local machine)
```bash
# Transfer files from slurm to local machine
RELATIVE_FILEPATH=outputs/rl_logs/TFV6_PPO_PARALLEL_SLURM_24232338/
rsync -avz --include="events.out.tfevents.*" --include="config.json" --exclude="*.pth" erikhbj@idun.hpc.ntnu.no:/cluster/home/erikhbj/master/master-spring-ad-lead/${RELATIVE_FILEPATH} ./${RELATIVE_FILEPATH}
```


# Analysis

## Debug viz folder to mp4
```bash
python3 rl_finetuning/mp4_convert_folder.py outputs/rl_logs/TFV6_PPO_PARALLEL_LOCAL_NEWROUTES1213_newTP_entropy0/latest/debug_viz
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
