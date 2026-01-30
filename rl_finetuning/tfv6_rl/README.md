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
