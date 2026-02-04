## Bench2Drive
From repo root in idun run: `bash slurm/experiments/001_example/020_b2d_0.sh` to create many slurm jobs for B2D.

`cd /outputs/evaluation/001_example/020_b2d_0/` for å finne resultater fra idun.

## Run dashboard from idun files
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

## SSH with vscode to 5090 computer
Connect to eduroam network (or VPN)
```bash
# Tailscale IP: erikhbj@100.92.150.98
# Trenger ikke være på VPN. Anbefalt ikke.
ssh erikhbj@100.92.150.98
```
Or go to vscode: `Ctrl+Shift+P> Remote-SSH: Connect to Host...`

## Smoke test PPO single update
Terminal 1:
```bash
bash ./scripts/clean_carla.sh
bash ./scripts/start_carla.sh
```
Terminal 2:
```bash
# Assumes carla server running
TFV6_RL_DEBUG=1 bash rl_finetuning/run_tfv6_smoke.sh outputs/checkpoints/tfv6_resnet34 outputs/rl_logs TFV6_PPO_SMOKE
```
