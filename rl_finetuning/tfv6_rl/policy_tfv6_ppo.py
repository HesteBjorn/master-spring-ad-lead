from __future__ import annotations

import json
import os

import torch
from torch import nn

from lead.inference.config_closed_loop import ClosedLoopConfig
from lead.tfv6.tfv6 import TFv6
from lead.training.config_training import TrainingConfig
from rl_finetuning.tfv6_rl.action_codec import ActionCodec, infer_action_head_usage
from rl_finetuning.tfv6_rl.gaussian_dist import (
    CorrelatedGaussianDistribution,
    DiagGaussianDistribution,
)


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
        rl_config=None,
        use_correlated_noise: bool = True,
        correlated_noise_rho: float = 0.8,
        noise_ramp: bool = True,
        train_planning_decoder_only: bool = True,
        start_log_std: float = -4.0,
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

        self.train_planning_decoder_only = train_planning_decoder_only

        # Match TFv6 inference behavior (autocast when mixed precision is enabled).
        self.autocast_dtype = self.training_config.torch_float_type
        self.autocast_enabled = (
            self.training_config.use_mixed_precision_training
            and self.device.type == "cuda"
        )

        closed_loop_config = ClosedLoopConfig(raise_error_on_missing_key=False)
        use_route, use_waypoints, use_target_speed = infer_action_head_usage(
            self.training_config,
            steer_modality=closed_loop_config.steer_modality,
            throttle_modality=closed_loop_config.throttle_modality,
            brake_modality=closed_loop_config.brake_modality,
        )
        self.action_codec = ActionCodec(
            self.training_config,
            use_route=use_route,
            use_waypoints=use_waypoints,
            use_target_speed=use_target_speed,
        )
        self.action_dim = self.action_codec.action_dim

        self.use_correlated_noise = use_correlated_noise
        self.correlated_noise_rho = correlated_noise_rho
        self.noise_ramp = noise_ramp
        self.skip_perception_heads = True
        self.log_std_min = -5.0
        self.log_std_max = 1.0
        if rl_config is not None:
            self.use_correlated_noise = bool(
                getattr(rl_config, "use_correlated_noise", self.use_correlated_noise)
            )
            self.correlated_noise_rho = float(
                getattr(rl_config, "correlated_noise_rho", self.correlated_noise_rho)
            )
            self.noise_ramp = bool(getattr(rl_config, "noise_ramp", self.noise_ramp))
            self.skip_perception_heads = bool(
                getattr(rl_config, "skip_perception_heads", self.skip_perception_heads)
            )
            self.log_std_min = float(
                getattr(rl_config, "log_std_min", self.log_std_min)
            )
            self.log_std_max = float(
                getattr(rl_config, "log_std_max", self.log_std_max)
            )
            self.train_planning_decoder_only = bool(
                getattr(
                    rl_config,
                    "train_planning_decoder_only",
                    self.train_planning_decoder_only,
                )
            )

        self._configure_trainable_tfv6_modules()

        if self.use_correlated_noise:
            self.action_dist = CorrelatedGaussianDistribution(
                self.action_codec, rho=self.correlated_noise_rho
            )
        else:
            self.action_dist = DiagGaussianDistribution(self.action_dim)
        self.log_std = nn.Parameter(start_log_std * torch.ones(self.action_dim))

        value_in_dim = self.training_config.transfuser_token_dim
        self.value_head = nn.Sequential(
            nn.Linear(value_in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

    def _set_module_trainable(self, module: nn.Module, trainable: bool) -> None:
        for param in module.parameters():
            param.requires_grad = trainable

    def _configure_active_planning_heads(self) -> None:
        """Freeze planning decoder heads that are not part of active PPO actions."""
        planning_decoder = getattr(self.tfv6, "planning_decoder", None)
        if planning_decoder is None:
            return

        if hasattr(planning_decoder, "route_decoder"):
            self._set_module_trainable(
                planning_decoder.route_decoder, self.action_codec.predict_route
            )
        if hasattr(planning_decoder, "wp_decoder"):
            self._set_module_trainable(
                planning_decoder.wp_decoder, self.action_codec.predict_waypoints
            )
        if hasattr(planning_decoder, "target_speed_decoder"):
            self._set_module_trainable(
                planning_decoder.target_speed_decoder,
                self.action_codec.predict_target_speed,
            )
        # NavSim-only heading head shares waypoint semantics.
        if hasattr(planning_decoder, "heading_decoder"):
            self._set_module_trainable(
                planning_decoder.heading_decoder, self.action_codec.predict_waypoints
            )

    def _configure_trainable_tfv6_modules(self) -> None:
        if self.train_planning_decoder_only:
            # Decoder-only RL finetuning: keep TFv6 frozen except planning decoder.
            self._set_module_trainable(self.tfv6, False)
            if hasattr(self.tfv6, "planning_decoder"):
                self._set_module_trainable(self.tfv6.planning_decoder, True)
                self._configure_active_planning_heads()
            return

        # Partial/full finetuning mode.
        self._set_module_trainable(self.tfv6, True)
        self._configure_active_planning_heads()

    def _build_value_features(self) -> torch.Tensor:
        kv = getattr(self.tfv6.planning_decoder, "kv", None)
        if kv is None:
            raise RuntimeError("Planning decoder context tokens not available.")
        return kv.mean(dim=1)

    def _apply_noise_ramp(self, log_std: torch.Tensor) -> torch.Tensor:
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        if not self.noise_ramp:
            return log_std
        mask = torch.ones(self.action_dim, device=log_std.device, dtype=log_std.dtype)
        if self.action_codec.predict_route and self.action_codec.num_route_points > 0:
            start = self.action_codec.slices.route.start
            mask[start : start + 2] = 0.5
        if self.action_codec.predict_waypoints and self.action_codec.num_waypoints > 0:
            start = self.action_codec.slices.waypoints.start
            mask[start : start + 2] = 0.5
        std = torch.exp(log_std) * mask
        return torch.log(std + 1e-6)

    def get_value(self, obs_dict, *_args, **_kwargs) -> torch.Tensor:
        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=self.autocast_dtype,
            enabled=self.autocast_enabled,
        ):
            _ = self.tfv6(obs_dict, skip_perception_heads=self.skip_perception_heads)
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
            predictions = self.tfv6(
                obs_dict, skip_perception_heads=self.skip_perception_heads
            )

        route = predictions.pred_route if self.action_codec.predict_route else None
        waypoints = (
            predictions.pred_future_waypoints
            if self.action_codec.predict_waypoints
            else None
        )
        target_speed = (
            predictions.pred_target_speed_scalar
            if self.action_codec.predict_target_speed
            else None
        )

        action_mean = self.action_codec.encode(route, waypoints, target_speed).float()
        log_std = self.log_std.unsqueeze(0).expand_as(action_mean)
        log_std = self._apply_noise_ramp(log_std)
        dist = self.action_dist.proba_distribution(action_mean, log_std)

        if actions is None:
            actions = dist.get_actions(sample_type)

        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        if entropy.ndim > 1:
            entropy = entropy.sum(1)

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
