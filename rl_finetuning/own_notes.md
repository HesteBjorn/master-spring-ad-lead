# Evaluation
## Bench2Drive Idun
From repo root in idun run: `bash slurm/experiments/001_example/020_b2d_0.sh` to create many slurm jobs for B2D.

`cd /outputs/evaluation/001_example/020_b2d_0/` for å finne resultater fra idun.

## Run b2d dashboard from idun files
```bash
# Login to idun and port forward
ssh -L 5000:localhost:5000 erikhbj@idun.hpc.ntnu.no
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

## Visualize losses and rewards
```bash
tensorboard --logdir outputs/rl_logs
```

## Long local run without watchdog (not reccommended)
```bash
# Assumes carla server running
bash rl_finetuning/train_long_local.sh \
  outputs/checkpoints/tfv6_resnet34 \
  outputs/rl_logs \
  TFV6_PPO_LONG_LOCAL \
  # outputs/rl_logs/TFV6_PPO_LONG_LOCAL/model_latest_000000123.pth  # Optional resume argument, pointing to model .pth
```

## Weekend run with auto-carla restarts
```bash
RUN_CONFIG_FILE=rl_finetuning/configs/weekend_local_run_stable.env \
bash rl_finetuning/run_tfv6_watchdog_overnight.sh \
  outputs/checkpoints/tfv6_resnet34 \
  outputs/rl_logs \
  TFV6_PPO_WEEKEND_20260214_correlationrho095_logstd45 \
  --debug-viz --debug-viz-every-n 1 --debug-viz-max-images 100  # Optional debug visualization
```

## Kill watchdog
```bash
pkill -f "rl_finetuning/run_tfv6_watchdog_overnight.sh|rl_finetuning/train_long_local.sh|rl_finetuning/train_tfv6_ppo.py|torchrun.*train_tfv6_ppo.py|custom_leaderboard/leaderboard/leaderboard/leaderboard_evaluator.py|CarlaUE4|CarlaUE4-Linux-Shipping|nvidia-smi  --query"
sleep 2
ps -ef | rg "run_tfv6_watchdog_overnight|train_long_local|train_tfv6_ppo|torchrun|leaderboard_evaluator|CarlaUE4|nvidia-smi --query" -i
```

## Parallell training
### Local parallell training
```bash
RUN_CONFIG_FILE=rl_finetuning/configs/weekend_local_run_stable.env \
bash rl_finetuning/train_tfv6_ppo_parallell.sh \
  outputs/checkpoints/tfv6_resnet34 \
  outputs/rl_logs \
  TFV6_PPO_PARALLEL_LOCAL \
```

### Slurm parallell training
TODO


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
