from __future__ import annotations

import json
import os

import torch
from torch import nn

from lead.inference.config_closed_loop import ClosedLoopConfig
from lead.tfv6.tfv6 import TFv6
from lead.training.config_training import TrainingConfig
from rl_finetuning.tfv6_rl.action_codec import ActionCodec, infer_action_head_usage
from rl_finetuning.tfv6_rl.action_noise_codec import ActionNoiseCodec


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
        self.log_std_init = -4.0
        self.log_std_min = -5.0
        self.log_std_max = 1.0
        self.action_noise_dist = "gaussian"
        self.route_sampling_technique = "spline_curvature_perturbation"
        self.heading_amplitude1_std_init = 0.07
        self.heading_amplitude2_std_init = 0.03
        self.path_std_base_frac = 0.15
        self.disable_learned_noise_head = True
        self.use_privileged_measurements = True
        self.num_privileged_measurements = 0
        self.use_kl_to_reference = True
        self.kl_to_reference_coef = 1e-4
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
            self.log_std_init = float(
                getattr(rl_config, "log_std_init", self.log_std_init)
            )
            self.log_std_min = float(
                getattr(rl_config, "log_std_min", self.log_std_min)
            )
            self.log_std_max = float(
                getattr(rl_config, "log_std_max", self.log_std_max)
            )
            self.action_noise_dist = str(
                getattr(rl_config, "action_noise_dist", self.action_noise_dist)
            )
            self.route_sampling_technique = str(
                getattr(
                    rl_config,
                    "route_sampling_technique",
                    self.route_sampling_technique,
                )
            )
            self.heading_amplitude1_std_init = float(
                getattr(
                    rl_config,
                    "heading_amplitude1_std_init",
                    self.heading_amplitude1_std_init,
                )
            )
            self.heading_amplitude2_std_init = float(
                getattr(
                    rl_config,
                    "heading_amplitude2_std_init",
                    self.heading_amplitude2_std_init,
                )
            )
            self.path_std_base_frac = float(
                getattr(rl_config, "path_std_base_frac", self.path_std_base_frac)
            )
            self.disable_learned_noise_head = bool(
                getattr(
                    rl_config,
                    "disable_learned_noise_head",
                    self.disable_learned_noise_head,
                )
            )
            self.use_privileged_measurements = bool(
                getattr(
                    rl_config,
                    "use_value_measurements",
                    self.use_privileged_measurements,
                )
            )
            self.num_privileged_measurements = int(
                getattr(
                    rl_config,
                    "num_value_measurements",
                    self.num_privileged_measurements,
                )
            )
            self.use_kl_to_reference = bool(
                getattr(rl_config, "use_kl_to_reference", self.use_kl_to_reference)
            )
            self.kl_to_reference_coef = float(
                getattr(rl_config, "kl_to_reference_coef", self.kl_to_reference_coef)
            )
            self.train_planning_decoder_only = bool(
                getattr(
                    rl_config,
                    "train_planning_decoder_only",
                    self.train_planning_decoder_only,
                )
            )

        self._configure_trainable_tfv6_modules()
        self.reference_tfv6 = None
        if self.use_kl_to_reference:
            self.reference_tfv6 = TFv6(self.device, self.training_config)
            ref_state_dict = torch.load(
                weights_path, map_location=self.device, weights_only=True
            )
            self.reference_tfv6.load_state_dict(ref_state_dict, strict=True)
            self.reference_tfv6.to(self.device)
            self.reference_tfv6.eval()
            for param in self.reference_tfv6.parameters():
                param.requires_grad_(False)
        self.action_noise_codec = ActionNoiseCodec(
            self.action_codec,
            distribution_type=self.action_noise_dist,
            sampling_technique=self.route_sampling_technique,
            use_correlated_gaussian=self.use_correlated_noise,
            correlated_noise_rho=self.correlated_noise_rho,
            noise_ramp=self.noise_ramp,
            log_std_min=self.log_std_min,
            log_std_max=self.log_std_max,
            heading_amplitude1_std_init=self.heading_amplitude1_std_init,
            heading_amplitude2_std_init=self.heading_amplitude2_std_init,
            path_std_base_frac=self.path_std_base_frac,
        )
        self.action_dist = self.action_noise_codec.action_dist
        self.privileged_obs_key = "privileged_measurements"
        self.value_token_dim = self.training_config.transfuser_token_dim
        value_in_dim = self.value_token_dim + (
            self.num_privileged_measurements if self.use_privileged_measurements else 0
        )
        # Predict grouped noise parameters with a small MLP head.
        # Hidden size matches the TFv6 planning token width for a minimal increase in capacity.
        self.action_noise_head = nn.Sequential(
            nn.Linear(value_in_dim, value_in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(value_in_dim, self.action_noise_codec.noise_pred_dim),
        )
        final_noise_layer = self.action_noise_head[-1]
        nn.init.zeros_(
            final_noise_layer.weight
        )  # Exact constant initialization; bias sets the initial exploration level.
        init_noise_bias = self.action_noise_codec.default_head_bias_vector(
            log_std_init=self.log_std_init,
            heading_amplitude1_std_init=self.heading_amplitude1_std_init,
            heading_amplitude2_std_init=self.heading_amplitude2_std_init,
            device=final_noise_layer.bias.device,
            dtype=final_noise_layer.bias.dtype,
        )
        with torch.no_grad():
            final_noise_layer.bias.copy_(init_noise_bias)

        # Constant-noise mode: keep std/head at initialized values.
        if self.disable_learned_noise_head:
            for param in self.action_noise_head.parameters():
                param.requires_grad_(False)

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

    def _extract_tfv6_obs(self, obs_dict: dict) -> dict:
        if self.privileged_obs_key not in obs_dict:
            return obs_dict
        return {k: v for k, v in obs_dict.items() if k != self.privileged_obs_key}

    def _get_privileged_measurements(
        self, obs_dict: dict, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        if (
            not self.use_privileged_measurements
            or self.num_privileged_measurements <= 0
        ):
            return torch.zeros((batch_size, 0), device=device, dtype=dtype)
        pm = obs_dict.get(self.privileged_obs_key, None)
        if pm is None:
            return torch.zeros(
                (batch_size, self.num_privileged_measurements),
                device=device,
                dtype=dtype,
            )
        pm = pm.to(device=device, dtype=dtype)
        if pm.ndim == 1:
            pm = pm.unsqueeze(0)
        if pm.shape[1] != self.num_privileged_measurements:
            raise ValueError(
                f"Expected {self.num_privileged_measurements} privileged measurements, got {pm.shape[1]}"
            )
        return pm

    def _build_value_and_noise_features(self, obs_dict: dict) -> torch.Tensor:
        value_features = self._build_value_features().float()
        privileged = self._get_privileged_measurements(
            obs_dict,
            batch_size=value_features.shape[0],
            device=value_features.device,
            dtype=value_features.dtype,
        )
        if privileged.shape[1] == 0:
            return value_features
        return torch.cat((value_features, privileged), dim=1)

    def predict_action_noise(self, value_features: torch.Tensor) -> torch.Tensor:
        return self.action_noise_head(value_features)

    @torch.no_grad()
    def get_reference_action_mean(self, obs_dict) -> torch.Tensor:
        if self.reference_tfv6 is None:
            raise RuntimeError("Reference TFv6 is not enabled.")
        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=self.autocast_dtype,
            enabled=self.autocast_enabled,
        ):
            ref_predictions = self.reference_tfv6(
                self._extract_tfv6_obs(obs_dict),
                skip_perception_heads=self.skip_perception_heads,
            )
        ref_route = (
            ref_predictions.pred_route if self.action_codec.predict_route else None
        )
        ref_waypoints = (
            ref_predictions.pred_future_waypoints
            if self.action_codec.predict_waypoints
            else None
        )
        ref_target_speed = (
            ref_predictions.pred_target_speed_scalar
            if self.action_codec.predict_target_speed
            else None
        )
        return self.action_codec.encode(
            ref_route, ref_waypoints, ref_target_speed
        ).float()

    def get_value(self, obs_dict, *_args, **_kwargs) -> torch.Tensor:
        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=self.autocast_dtype,
            enabled=self.autocast_enabled,
        ):
            _ = self.tfv6(
                self._extract_tfv6_obs(obs_dict),
                skip_perception_heads=self.skip_perception_heads,
            )
        value_features = self._build_value_and_noise_features(obs_dict)
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
                self._extract_tfv6_obs(obs_dict),
                skip_perception_heads=self.skip_perception_heads,
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
        value_features = self._build_value_and_noise_features(obs_dict)
        noise_pred = self.predict_action_noise(value_features)
        dist, noise_diag = self.action_noise_codec.proba_distribution(
            action_mean, noise_pred
        )

        if actions is None:
            actions = self.action_noise_codec.sample_actions(
                action_mean,
                noise_diag,
                dist,
                sample_type=sample_type,
            )

        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        if entropy.ndim > 1:
            entropy = entropy.sum(1)

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
            noise_diag.noise_pred.detach(),
            dist.distribution,
            None,
            noise_diag.diag_log_std.detach(),
            lstm_state,
        )
