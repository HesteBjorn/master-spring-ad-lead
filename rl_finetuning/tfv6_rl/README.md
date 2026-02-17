# TFv6 PPO Finetuning (plan-as-action)

This folder contains the TFv6 PPO integration used for plan-level RL finetuning.

## Components
- `env_agent_tfv6.py`: Leaderboard agent that builds TFv6 inputs, sends obs to PPO trainer, and applies plan actions.
- `env_gym_tfv6.py`: Gym env used by PPO trainer, receives obs via ZMQ.
- `policy_tfv6_ppo.py`: TFv6 PPO policy wrapper (planning decoder outputs → PPO actions).
- `action_codec.py`: Action vector ↔ (route, waypoints, target_speed).
- `obs_codec.py`: Observation schema and packing.

## Expected usage (debug)
1. Run CARLA and the custom leaderboard (same as CaRL), but use the new agent:
   - `--agent /path/to/repo/rl_finetuning/tfv6_rl/env_agent_tfv6.py`
   - `--agent-config /path/to/repo/outputs/checkpoints/tfv6_resnet34`

2. Run PPO trainer:
   ```bash
   torch.distributed.run --nnodes=1 --nproc_per_node=1 --max_restarts=0 \
     --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
     /path/to/repo/rl_finetuning/train_tfv6_ppo.py \
     --tcp_store_port 7000 \
     --logdir /path/to/logs \
     --exp_name TFV6_PPO_DEBUG \
     --ports 5555 \
     --total_batch_size 512 \
     --total_minibatch_size 128 \
     --update_epochs 3 \
     --total_timesteps 1000000 \
     --reward_type simple_reward \
     --tfv6_checkpoint /path/to/repo/outputs/checkpoints/tfv6_resnet34 \
     --debug_shapes 1
   ```

## One-command smoke test (short rollout)

This mirrors CaRL's training-debug flow (leaderboard process + PPO process), but keeps
the rollout intentionally tiny so you can quickly verify gradients/logging.

1. Start a CARLA server in a separate terminal (required):
   ```bash
   bash scripts/start_carla.sh
   ```
2. Run:
   ```bash
   bash rl_finetuning/run_tfv6_smoke.sh \
     outputs/checkpoints/tfv6_resnet34 \
     outputs/rl_logs \
     TFV6_PPO_SMOKE
   ```

Defaults are `total_batch_size=32`, `total_timesteps=32`, `update_epochs=1`.
If this succeeds, TensorBoard should show at least one PPO update (not just step 0 hyperparameters).

## Long local run (single launcher)

Use `train_long_local.sh` for both longer local debug runs and overnight runs.

```bash
TFV6_RL_DEBUG=1 bash rl_finetuning/train_long_local.sh \
  outputs/checkpoints/tfv6_resnet34 \
  outputs/rl_logs \
  TFV6_PPO_LONG_LOCAL
```

You can resume from a checkpoint by passing a 4th argument:
```bash
TFV6_RL_DEBUG=1 bash rl_finetuning/train_long_local.sh \
  outputs/checkpoints/tfv6_resnet34 \
  outputs/rl_logs \
  TFV6_PPO_LONG_LOCAL \
  outputs/rl_logs/TFV6_PPO_LONG_LOCAL/model_latest_000000xxx.pth
```

Use only:
- `rl_finetuning/train_long_local.sh`

Default profile is `town03_debug` (scenario route with repetitions). You can switch:

- `ROUTE_PROFILE=debug_suite`: merges selected files from `debug_routes_with_scenarios`.
  - Town filter via `DEBUG_SUITE_TOWNS` (default: `Town01..Town06`).
- `ROUTE_PROFILE=training`: uses `routes_training.xml` directly.

All key runtime knobs (`TOTAL_TIMESTEPS`, `TOTAL_BATCH_SIZE`, `REPETITIONS`, `TRACK`, etc.)
are exposed as environment variables in the script.

3. Optional debug:
   - Set `TFV6_RL_DEBUG=1` in the environment to enable extra logging in the agent.

## Parallel TFv6 PPO training (CaRL-style)

For robust multi-environment training with CARLA restart handling (CaRL-like flow), use:

```bash
RUN_CONFIG_FILE=rl_finetuning/configs/weekend_local_run_stable.env \
bash rl_finetuning/train_tfv6_ppo_parallell.sh \
  outputs/checkpoints/tfv6_resnet34 \
  outputs/rl_logs \
  TFV6_PPO_PARALLEL_LOCAL \
```

This launcher starts:
- one CARLA server + leaderboard client per environment
- distributed trainer workers via `torchrun`
- auto-resume from latest `model_latest_*.pth`
- crash monitoring + full process-group restarts (matching CaRL `train_parallel.py` behavior)

Configuration is controlled via env vars or `RUN_CONFIG_FILE`, e.g.:
- `NUM_ENVS_PER_NODE`
- `NUM_ENVS_PER_GPU`
- `GPU_IDS`
- `TRAIN_TOWNS`
- `ROUTES_FOLDER`
- `TRAINER_EXTRA_ARGS`

## Notes
- Observation schema and action layout are derived from the TFv6 checkpoint config.
- Action vector includes route, waypoints, and target speed (the activated planning heads).
- PPO uses a diagonal Gaussian distribution in normalized action space [-1, 1].

## Dry-run sanity check (no CARLA required)
```bash
python -m rl_finetuning.tfv6_rl.dry_run \
  --checkpoint /path/to/repo/outputs/checkpoints/tfv6_resnet34 \
  --batch-size 2 \
  --sample-type mean
```

## Aggregated Speed Distribution Shift Analysis

Compares target-speed scalar predictions from:
- `TF_v6` (old/base checkpoint)
- `Finetuned Policy` (finetuned checkpoint)

across all discovered route frames under a data root, using the same real-observation
construction path as dry-run (`build_real_obs`).

Run:
```bash
python rl_finetuning/analyze_speed_distribution_shift.py \
  --old-checkpoint outputs/checkpoints/tfv6_resnet34/model_0030_0.pth \
  --finetuned-checkpoint outputs/rl_logs/TFV6_PPO_WEEKEND_20260212_linear/model_latest_000000065.pth \
  --data-root data/carla_leaderboard2/data \
  --output-dir outputs/local_evaluation \
  --output-stem speed_distribution_shift_all_routes
```

Outputs:
- Plot: `outputs/local_evaluation/speed_distribution_shift_all_routes.png`
- Aggregate stats: `outputs/local_evaluation/speed_distribution_shift_all_routes.txt`
- Per-route CSV: `outputs/local_evaluation/speed_distribution_shift_all_routes_per_route.csv`
