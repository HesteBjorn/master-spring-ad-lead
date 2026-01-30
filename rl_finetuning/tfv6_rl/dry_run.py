from __future__ import annotations

import argparse

import numpy as np
import torch

from lead.common.constants import RadarDataIndex
from rl_finetuning.tfv6_rl.action_codec import ActionCodec
from rl_finetuning.tfv6_rl.obs_codec import ObsCodec
from rl_finetuning.tfv6_rl.policy_tfv6_ppo import TFv6PPOPolicy, load_training_config


def build_dummy_obs(
    obs_codec: ObsCodec,
    training_config,
    batch_size: int,
    device: torch.device,
):
    obs = {}
    for spec in obs_codec.specs:
        shape = (batch_size,) + spec.shape
        if spec.key == "rgb":
            arr = np.random.randint(0, 255, size=shape, dtype=np.uint8)
        elif spec.key == "rasterized_lidar":
            arr = np.random.rand(*shape).astype(np.float32)
        elif spec.key == "radar":
            # Radar schema: [x, y, z, v, sensor_id]
            arr = np.zeros(shape, dtype=np.float32)
            arr[..., RadarDataIndex.X] = np.random.uniform(
                training_config.min_x_meter,
                training_config.max_x_meter,
                size=shape[:-1],
            )
            arr[..., RadarDataIndex.Y] = np.random.uniform(
                training_config.min_y_meter,
                training_config.max_y_meter,
                size=shape[:-1],
            )
            arr[..., RadarDataIndex.V] = np.random.uniform(
                0.0, training_config.max_speed, size=shape[:-1]
            )
            sensor_ids = np.random.randint(
                0,
                training_config.num_radar_sensors,
                size=shape[:-1],
                dtype=np.int64,
            )
            arr[..., RadarDataIndex.SENSOR_ID] = sensor_ids.astype(np.float32)
        elif spec.key in ("command", "next_command"):
            arr = np.zeros(shape, dtype=np.float32)
            # set a random one-hot command per batch
            for b in range(batch_size):
                idx = np.random.randint(0, 6)
                arr[b, idx] = 1.0
        else:
            arr = np.random.randn(*shape).astype(np.float32)
        obs[spec.key] = torch.tensor(arr, device=device)
    return obs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", required=True, type=str, help="Path to TFv6 checkpoint folder"
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--sample-type", type=str, default="mean", choices=["mean", "sample"]
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    training_config = load_training_config(args.checkpoint)
    obs_codec = ObsCodec(training_config)
    action_codec = ActionCodec(training_config)

    policy = TFv6PPOPolicy(
        observation_space=None,
        action_space=None,
        tfv6_checkpoint=args.checkpoint,
        tfv6_prefix="model",
        device=device,
    ).to(device)

    obs = build_dummy_obs(obs_codec, training_config, args.batch_size, device)

    with torch.no_grad():
        actions, logprob, entropy, values, _, mu, sigma, _, _, _, _ = policy.forward(
            obs, sample_type=args.sample_type
        )

    print("[dry_run] batch_size", args.batch_size)
    print("[dry_run] obs keys", list(obs.keys()))
    print("[dry_run] action_dim", actions.shape[-1])
    print("[dry_run] actions shape", tuple(actions.shape))
    print("[dry_run] logprob shape", tuple(logprob.shape))
    print("[dry_run] values shape", tuple(values.shape))

    route, waypoints, target_speed = action_codec.decode(actions.cpu())
    if route is not None:
        print("[dry_run] route shape", route.shape)
    if waypoints is not None:
        print("[dry_run] waypoints shape", waypoints.shape)
    if target_speed is not None:
        print("[dry_run] target_speed shape", target_speed.shape)

    print("[dry_run] OK")


if __name__ == "__main__":
    main()
