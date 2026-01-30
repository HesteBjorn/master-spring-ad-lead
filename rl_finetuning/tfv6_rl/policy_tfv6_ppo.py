from __future__ import annotations

import json
import os

import torch
from torch import nn

from lead.tfv6.tfv6 import TFv6
from lead.training.config_training import TrainingConfig
from rl_finetuning.tfv6_rl.action_codec import ActionCodec
from rl_finetuning.tfv6_rl.gaussian_dist import DiagGaussianDistribution


def load_training_config(checkpoint_dir: str) -> TrainingConfig:
    config_path = os.path.join(checkpoint_dir, "config.json")
    with open(config_path, encoding="utf-8") as f:
        loaded = json.load(f)
    return TrainingConfig(loaded, raise_error_on_missing_key=False)


def find_model_file(checkpoint_dir: str, prefix: str = "model") -> str:
    files = [
        f
        for f in os.listdir(checkpoint_dir)
        if f.startswith(prefix) and f.endswith(".pth")
    ]
    if not files:
        raise FileNotFoundError(
            f"No model weights found in {checkpoint_dir} with prefix '{prefix}'"
        )
    files.sort()
    return os.path.join(checkpoint_dir, files[-1])


class TFv6PPOPolicy(nn.Module):
    """PPO policy wrapper around TFv6 planning decoder outputs."""

    def __init__(
        self,
        observation_space,
        action_space,
        tfv6_checkpoint: str,
        tfv6_prefix: str = "model",
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.training_config = load_training_config(tfv6_checkpoint)
        self.tfv6 = TFv6(self.device, self.training_config)
        weights_path = find_model_file(tfv6_checkpoint, prefix=tfv6_prefix)
        state_dict = torch.load(
            weights_path, map_location=self.device, weights_only=True
        )
        self.tfv6.load_state_dict(state_dict, strict=True)
        self.tfv6.to(self.device)

        # Match TFv6 inference behavior (autocast when mixed precision is enabled).
        self.autocast_dtype = self.training_config.torch_float_type
        self.autocast_enabled = (
            self.training_config.use_mixed_precision_training
            and self.device.type == "cuda"
        )

        self.action_codec = ActionCodec(self.training_config)
        self.action_dim = self.action_codec.action_dim

        self.action_dist = DiagGaussianDistribution(self.action_dim)
        self.log_std = nn.Parameter(-1.0 * torch.ones(self.action_dim))

        value_in_dim = self.training_config.transfuser_token_dim
        self.value_head = nn.Sequential(
            nn.Linear(value_in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

    def _build_value_features(self) -> torch.Tensor:
        kv = getattr(self.tfv6.planning_decoder, "kv", None)
        if kv is None:
            raise RuntimeError("Planning decoder context tokens not available.")
        return kv.mean(dim=1)

    def get_value(self, obs_dict, *_args, **_kwargs) -> torch.Tensor:
        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=self.autocast_dtype,
            enabled=self.autocast_enabled,
        ):
            _ = self.tfv6(obs_dict)
        value_features = self._build_value_features().float()
        return self.value_head(value_features)

    def forward(
        self,
        obs_dict,
        actions=None,
        sample_type: str = "sample",
        exploration_suggests=None,
        lstm_state=None,
        done=None,
    ) -> tuple:
        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=self.autocast_dtype,
            enabled=self.autocast_enabled,
        ):
            predictions = self.tfv6(obs_dict)

        route = (
            predictions.pred_route
            if self.training_config.predict_spatial_path
            else None
        )
        waypoints = (
            predictions.pred_future_waypoints
            if self.training_config.predict_temporal_spatial_waypoints
            else None
        )
        target_speed = (
            predictions.pred_target_speed_scalar
            if self.training_config.predict_target_speed
            else None
        )

        action_mean = self.action_codec.encode(route, waypoints, target_speed).float()
        log_std = self.log_std.unsqueeze(0).expand_as(action_mean)
        dist = self.action_dist.proba_distribution(action_mean, log_std)

        if actions is None:
            actions = dist.get_actions(sample_type)

        log_prob = dist.log_prob(actions)
        entropy = dist.entropy().sum(1)

        value_features = self._build_value_features().float()
        values = self.value_head(value_features)

        exp_loss = None
        if exploration_suggests is not None:
            exp_loss = dist.exploration_loss(exploration_suggests)

        return (
            actions,
            log_prob,
            entropy,
            values,
            exp_loss,
            action_mean.detach(),
            log_std.detach(),
            dist.distribution,
            None,
            None,
            lstm_state,
        )
