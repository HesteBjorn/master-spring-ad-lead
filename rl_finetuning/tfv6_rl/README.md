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

## Longer smoke/debug run (~1 hour)

Use this when you want more realistic rollouts and enough PPO updates to inspect
reward/loss trends in TensorBoard.

```bash
TFV6_RL_DEBUG=1 bash rl_finetuning/run_tfv6_smoke_long.sh \
  outputs/checkpoints/tfv6_resnet34 \
  outputs/rl_logs \
  TFV6_PPO_SMOKE_LONG
```

You can resume from a checkpoint by passing a 4th argument:
```bash
TFV6_RL_DEBUG=1 bash rl_finetuning/run_tfv6_smoke_long.sh \
  outputs/checkpoints/tfv6_resnet34 \
  outputs/rl_logs \
  TFV6_PPO_SMOKE_LONG \
  outputs/rl_logs/TFV6_PPO_SMOKE_LONG/model_latest_000000xxx.pth
```

The overnight script accepts the same optional 4th argument:
```bash
TFV6_RL_DEBUG=1 bash rl_finetuning/run_tfv6_smoke_overnight.sh \
  outputs/checkpoints/tfv6_resnet34 \
  outputs/rl_logs \
  TFV6_PPO_SMOKE_OVERNIGHT \
  outputs/rl_logs/TFV6_PPO_SMOKE_OVERNIGHT/model_latest_000000xxx.pth
```

Default profile is `town03_debug` (scenario route with repetitions). You can switch:

- `ROUTE_PROFILE=debug_suite`: merges selected files from `debug_routes_with_scenarios`.
  - Town filter via `DEBUG_SUITE_TOWNS` (default: `Town01..Town06`).
- `ROUTE_PROFILE=training`: uses `routes_training.xml` directly.

All key runtime knobs (`TOTAL_TIMESTEPS`, `TOTAL_BATCH_SIZE`, `REPETITIONS`, `TRACK`, etc.)
are exposed as environment variables in the script.

3. Optional debug:
   - Set `TFV6_RL_DEBUG=1` in the environment to enable extra logging in the agent.

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
