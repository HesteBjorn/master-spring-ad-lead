from __future__ import annotations

import json
import os

import torch
from torch import nn

from lead.inference.config_closed_loop import ClosedLoopConfig
from lead.tfv6.tfv6 import TFv6
from lead.training.config_training import TrainingConfig
from rl_finetuning.tfv6_rl.action_codec import ActionCodec, infer_action_head_usage
from rl_finetuning.tfv6_rl.action_noise_codec import (
    ActionNoiseCodec,
    DiagGaussianDistribution,
    build_route_basis,
)
from rl_finetuning.tfv6_rl.privileged_measurements import privileged_measurement_dim


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
        self.log_std_init_route = None
        self.log_std_init_speed = None
        self.log_std_min = -5.0
        self.log_std_max = 1.0
        self.speed_sampling_style = "native_categorical"
        self.action_noise_dist = "gaussian"
        self.route_sampling_technique = "spline_curvature_perturbation"
        self.heading_amplitude1_std_init = 0.07
        self.heading_amplitude2_std_init = 0.03
        self.path_std_base_frac = 0.15
        self.lowrank_route_rank = 6
        self.lowrank_route_std_init = 0.015
        self.disable_learned_noise_head = True
        self.speed_temperature = 1.0
        self.critic_updates_shared_features = True
        self.use_privileged_measurements = True
        self.num_privileged_measurements = 0
        self.use_kl_to_reference = True
        self.kl_to_reference_coef = 1e-4
        self.use_residual_policy = False
        self.residual_route_rank = 2
        self.residual_alpha = 0.15
        self.residual_alpha_speed = 0.15
        self.disable_residual_route = False
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
            self.log_std_init_route = getattr(
                rl_config, "log_std_init_route", self.log_std_init_route
            )
            if self.log_std_init_route is not None:
                self.log_std_init_route = float(self.log_std_init_route)
            self.log_std_init_speed = getattr(
                rl_config, "log_std_init_speed", self.log_std_init_speed
            )
            if self.log_std_init_speed is not None:
                self.log_std_init_speed = float(self.log_std_init_speed)
            self.log_std_min = float(
                getattr(rl_config, "log_std_min", self.log_std_min)
            )
            self.log_std_max = float(
                getattr(rl_config, "log_std_max", self.log_std_max)
            )
            self.speed_sampling_style = str(
                getattr(
                    rl_config,
                    "speed_sampling_style",
                    getattr(
                        rl_config,
                        "speed_samping_style",
                        self.speed_sampling_style,
                    ),
                )
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
            self.lowrank_route_rank = int(
                getattr(rl_config, "lowrank_route_rank", self.lowrank_route_rank)
            )
            self.lowrank_route_std_init = float(
                getattr(
                    rl_config,
                    "lowrank_route_std_init",
                    self.lowrank_route_std_init,
                )
            )
            self.disable_learned_noise_head = bool(
                getattr(
                    rl_config,
                    "disable_learned_noise_head",
                    self.disable_learned_noise_head,
                )
            )
            self.speed_temperature = float(
                getattr(rl_config, "speed_temperature", self.speed_temperature)
            )
            self.critic_updates_shared_features = bool(
                getattr(
                    rl_config,
                    "critic_updates_shared_features",
                    self.critic_updates_shared_features,
                )
            )
            self.use_privileged_measurements = bool(
                getattr(
                    rl_config,
                    "use_value_measurements",
                    self.use_privileged_measurements,
                )
            )
            if self.use_privileged_measurements:
                self.num_privileged_measurements = privileged_measurement_dim(rl_config)
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
            self.use_residual_policy = bool(
                getattr(rl_config, "use_residual_policy", self.use_residual_policy)
            )
            self.residual_route_rank = int(
                getattr(rl_config, "residual_route_rank", self.residual_route_rank)
            )
            self.residual_alpha = float(
                getattr(rl_config, "residual_alpha", self.residual_alpha)
            )
            self.residual_alpha_speed = float(
                getattr(rl_config, "residual_alpha_speed", self.residual_alpha_speed)
            )
            self.disable_residual_route = bool(
                getattr(
                    rl_config, "disable_residual_route", self.disable_residual_route
                )
            )
        if self.use_residual_policy:
            # Architectural constraint: the frozen base model replaces the KL trust region.
            self.use_kl_to_reference = False
        if self.speed_sampling_style == "mean_gassian":
            self.speed_sampling_style = "mean_gaussian"
        if self.speed_sampling_style not in ("mean_gaussian", "native_categorical"):
            raise ValueError(
                "speed_sampling_style must be one of "
                "{'mean_gaussian', 'native_categorical'}"
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
            lowrank_route_rank=self.lowrank_route_rank,
            lowrank_route_std_init=self.lowrank_route_std_init,
            speed_sampling_style=self.speed_sampling_style,
            speed_temperature=self.speed_temperature,
        )
        self.action_dist = self.action_noise_codec.action_dist
        self.privileged_obs_key = "privileged_measurements"
        self.value_token_dim = self.training_config.transfuser_token_dim
        # Noise head input: mean-pooled kv (token_dim) + privileged measurements.
        noise_in_dim = self.value_token_dim + (
            self.num_privileged_measurements if self.use_privileged_measurements else 0
        )
        # Noise head: predicts per-group log_std for the action distribution (normal mode only).
        # In residual mode this module is a 1-output dummy — the residual head handles
        # both means and log_stds for the basis-coefficient distribution.
        noise_head_out_dim = (
            1 if self.use_residual_policy else self.action_noise_codec.noise_pred_dim
        )
        self.action_noise_head = nn.Sequential(
            nn.Linear(noise_in_dim, noise_in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(noise_in_dim, noise_head_out_dim),
        )
        final_noise_layer = self.action_noise_head[-1]
        nn.init.zeros_(final_noise_layer.weight)
        if not self.use_residual_policy:
            init_noise_bias = self.action_noise_codec.default_head_bias_vector(
                log_std_init=self.log_std_init,
                log_std_init_route=self.log_std_init_route,
                log_std_init_speed=self.log_std_init_speed,
                heading_amplitude1_std_init=self.heading_amplitude1_std_init,
                heading_amplitude2_std_init=self.heading_amplitude2_std_init,
                device=final_noise_layer.bias.device,
                dtype=final_noise_layer.bias.dtype,
            )
            with torch.no_grad():
                final_noise_layer.bias.copy_(init_noise_bias)

        # Constant-noise mode: keep std/head at initialized values (normal mode only).
        if self.disable_learned_noise_head:
            for param in self.action_noise_head.parameters():
                param.requires_grad_(False)

        # Residual RL components (only when use_residual_policy=True).
        self.residual_head = None
        self._last_base_action_mean: torch.Tensor | None = None
        if self.use_residual_policy:
            if not self.action_codec.predict_route:
                raise ValueError(
                    "Residual policy requires route predictions to be enabled."
                )
            if not self.action_codec.predict_target_speed:
                raise ValueError(
                    "Residual policy requires target speed predictions to be enabled."
                )

            # Fixed route correction basis: [route_dim, residual_route_rank]
            route_basis = build_route_basis(
                num_route_points=self.action_codec.num_route_points,
                route_dim=self.action_codec.route_dim,
                rank=self.residual_route_rank,
            )
            self.register_buffer("residual_route_basis", route_basis, persistent=False)

            # Normalized speed bin values for computing E[categorical speed].
            speed_bins = torch.tensor(
                self.training_config.target_speed_classes, dtype=torch.float32
            ) / float(self.training_config.max_speed)
            self.register_buffer("residual_speed_bins", speed_bins, persistent=False)

            # Residual head: outputs 2*(rank+1) values.
            # Layout: [n_0..n_{rank-1}, speed_mean,  log_σ_0..log_σ_{rank-1}, log_speed_σ]
            #          ←——— rank+1 means ———→           ←———— rank+1 log_stds ————→
            #
            # Input = attention-pooled KV tokens (skip + 4-query, always detached from
            # TFv6) + frozen base action (route_norm ++ speed_expected).
            # No privileged measurements — those are for the critic only.
            # KV always detached: the actor must never backpropagate into TFv6.
            residual_out_dim = 2 * (self.residual_route_rank + 1)
            self.num_residual_queries = 4
            self.residual_queries = nn.Parameter(
                torch.randn(self.num_residual_queries, self.value_token_dim) * 0.02
            )
            residual_spatial_dim = (
                1 + self.num_residual_queries
            ) * self.value_token_dim
            # Base action: route_dim route coords + 1 speed value (normalized, from frozen TFv6).
            residual_action_dim = self.action_codec.route_dim + 1
            residual_in_dim = residual_spatial_dim + residual_action_dim
            self.residual_head = nn.Sequential(
                nn.LayerNorm(residual_in_dim),
                nn.Linear(residual_in_dim, 256),
                nn.LayerNorm(256),
                nn.ReLU(inplace=True),
                nn.Linear(256, residual_out_dim),
            )
            route_log_std_init = (
                self.log_std_init_route
                if self.log_std_init_route is not None
                else self.log_std_init
            )
            speed_log_std_init = (
                self.log_std_init_speed
                if self.log_std_init_speed is not None
                else self.log_std_init
            )
            nn.init.zeros_(self.residual_head[-1].weight)
            with torch.no_grad():
                nn.init.zeros_(self.residual_head[-1].bias)
                # Route log_stds at indices [rank+1 : 2*rank+1]
                self.residual_head[-1].bias[
                    self.residual_route_rank + 1 : 2 * self.residual_route_rank + 1
                ] = route_log_std_init
                # Speed log_std at index [2*rank+1]
                self.residual_head[-1].bias[2 * self.residual_route_rank + 1] = (
                    speed_log_std_init
                )

            # Fixed log_std values used when disable_learned_noise_head=True.
            # Mirrors the normal-path behaviour of freezing the noise head.
            self._residual_fixed_route_log_std = float(
                self.log_std_init_route
                if self.log_std_init_route is not None
                else self.log_std_init
            )
            self._residual_fixed_speed_log_std = float(
                self.log_std_init_speed
                if self.log_std_init_speed is not None
                else self.log_std_init
            )

            # Reusable distribution object (no parameters, safe to keep as attribute).
            self._residual_dist = DiagGaussianDistribution()

        # Critic: mean-pool skip connection + 4-query attention pooling over kv tokens.
        # Mean-pool provides a stable direct gradient path (skip connection) so the MLP
        # can bootstrap immediately. The 4 attention queries learn specialized residual
        # views on top, each focusing on a different aspect of the scene.
        # Combined spatial dim: token_dim (mean) + num_queries * token_dim (attention).
        self.num_value_queries = 4
        self.value_queries = nn.Parameter(
            torch.randn(self.num_value_queries, self.value_token_dim) * 0.02
        )
        critic_spatial_dim = (1 + self.num_value_queries) * self.value_token_dim
        critic_in_dim = critic_spatial_dim + (
            self.num_privileged_measurements if self.use_privileged_measurements else 0
        )
        self.value_head = nn.Sequential(
            nn.LayerNorm(critic_in_dim),
            nn.Linear(critic_in_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

    @property
    def noise_pred_dim(self) -> int:
        """Dimension of the noise_pred tensor returned by forward().

        In residual mode: 2*(rank+1) — the full residual head output
        (means + log_stds for the basis-coefficient distribution).
        In normal mode: the codec's noise_pred_dim.
        """
        if self.use_residual_policy:
            return 2 * (self.residual_route_rank + 1)
        return self.action_noise_codec.noise_pred_dim

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
        if self.use_residual_policy:
            # Residual RL: entire TFv6 is frozen. Only the residual head is trained.
            self._set_module_trainable(self.tfv6, False)
            return

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
        """Mean-pooled kv tokens — used by the noise head."""
        kv = getattr(self.tfv6.planning_decoder, "kv", None)
        if kv is None:
            raise RuntimeError("Planning decoder context tokens not available.")
        return kv.mean(dim=1)

    def _build_critic_spatial_features(self) -> torch.Tensor:
        """Mean-pool skip + 4-query attention pooling over kv tokens for the critic.

        The mean-pool term is a skip connection: it gives the MLP a direct,
        stable gradient path that bypasses the attention softmax, so the critic
        can bootstrap immediately while the attention queries slowly specialise.
        The attention queries each attend independently over all tokens, learning
        residual specialised views on top of the mean-pool baseline.

        Output shape: [B, (1 + num_queries) * token_dim]

        Detach respects critic_updates_shared_features: kv is stopped before
        the attention so TFv6 receives no gradient from the critic, while the
        query parameters still receive gradient via the attention weights.
        """
        kv = getattr(self.tfv6.planning_decoder, "kv", None)
        if kv is None:
            raise RuntimeError("Planning decoder context tokens not available.")
        kv_f = kv if self.critic_updates_shared_features else kv.detach()
        # Skip connection: mean-pool over all tokens — direct gradient path to MLP.
        mean_pooled = kv_f.mean(dim=1)  # [B, token_dim]
        scale = self.value_token_dim**0.5
        # scores: [B, num_queries, N_tokens]
        scores = torch.einsum("qd,bnd->bqn", self.value_queries, kv_f) / scale
        weights = scores.softmax(dim=-1)
        # attention_pooled: [B, num_queries, token_dim]
        attention_pooled = torch.einsum("bqn,bnd->bqd", weights, kv_f)
        # Concatenate skip (mean) and attention along feature dim → [B, (1+Q)*D]
        return torch.cat(
            [mean_pooled, attention_pooled.flatten(start_dim=1)], dim=1
        ).float()

    def _build_residual_spatial_features(self) -> torch.Tensor:
        """Mean-pool skip + query attention over KV tokens for the residual head.

        Mirrors _build_critic_spatial_features but uses the separate residual_queries
        and always detaches KV — the actor must never propagate gradients into the
        frozen TFv6 backbone.

        Output shape: [B, (1 + num_residual_queries) * token_dim]
        """
        kv = getattr(self.tfv6.planning_decoder, "kv", None)
        if kv is None:
            raise RuntimeError("Planning decoder context tokens not available.")
        kv_d = kv.detach()  # always detach — actor must not update TFv6
        mean_pooled = kv_d.mean(dim=1)  # [B, token_dim]
        scale = self.value_token_dim**0.5
        scores = torch.einsum("qd,bnd->bqn", self.residual_queries, kv_d) / scale
        weights = scores.softmax(dim=-1)
        attention_pooled = torch.einsum("bqn,bnd->bqd", weights, kv_d)
        return torch.cat(
            [mean_pooled, attention_pooled.flatten(start_dim=1)], dim=1
        ).float()

    def _build_critic_input(self, obs_dict: dict) -> torch.Tensor:
        """Critic input: attention-pooled kv features + privileged measurements."""
        spatial = self._build_critic_spatial_features()
        privileged = self._get_privileged_measurements(
            obs_dict,
            batch_size=spatial.shape[0],
            device=spatial.device,
            dtype=spatial.dtype,
        )
        if privileged.shape[1] == 0:
            return spatial
        return torch.cat((spatial, privileged), dim=1)

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

    def _critic_input_features(self, value_features: torch.Tensor) -> torch.Tensor:
        if self.critic_updates_shared_features:
            return value_features
        return value_features.detach()

    def predict_action_noise(self, value_features: torch.Tensor) -> torch.Tensor:
        return self.action_noise_head(value_features)

    def _forward_residual(
        self,
        obs_dict,
        actions=None,
        sample_type: str = "sample",
        lstm_state=None,
    ) -> tuple:
        """Forward pass for residual RL mode.

        The frozen TFv6 provides the base route and speed prediction.  The
        residual head predicts per-basis-mode means and log_stds, defining a
        diagonal Gaussian over the (rank+1) correction coefficients.

        At rollout:
          coeffs ~ N(means, exp(log_stds))    shape [B, rank+1]
          delta_route = alpha * route_coeffs @ basis.T
          delta_speed = alpha_speed * speed_coeff
          action = [route_base + delta_route, speed_base + delta_speed]

        At update (actions provided):
          coefficients are recovered by projecting the stored action back into
          basis space (valid because TFv6 is frozen → same base given same obs),
          and log_prob is computed in that (rank+1)-dim coefficient space.

        Residual head output layout (2*(rank+1) values):
          [n_0..n_{rank-1}, speed_mean,  log_σ_0..log_σ_{rank-1}, log_speed_σ]
        """
        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=self.autocast_dtype,
            enabled=self.autocast_enabled,
        ):
            predictions = self.tfv6(
                self._extract_tfv6_obs(obs_dict),
                skip_perception_heads=self.skip_perception_heads,
            )

        # --- Base route (normalized) — frozen TFv6 prediction ---
        route = predictions.pred_route
        assert route is not None, "Residual policy requires pred_route from TFv6."
        route = route.float()
        B = route.shape[0]
        route_scale = self.action_codec.route_scale.to(
            device=route.device, dtype=route.dtype
        )
        route_norm = (route / route_scale).reshape(
            B, -1
        )  # (B, route_dim) no clamp: keep exact base

        # --- Base speed: E[categorical] in normalized units ---
        assert predictions.pred_target_speed_distribution is not None, (
            "Residual policy requires pred_target_speed_distribution from TFv6."
        )
        target_speed_logits = (
            predictions.pred_target_speed_distribution.float()
        )  # (B, bins)
        speed_probs = torch.softmax(
            target_speed_logits / self.speed_temperature, dim=-1
        )
        speed_bins = self.residual_speed_bins.to(
            device=speed_probs.device, dtype=speed_probs.dtype
        )
        speed_expected = (speed_probs * speed_bins).sum(dim=-1, keepdim=True)  # (B, 1)

        # Store base action for debug visualization (readable by training loop after forward).
        self._last_base_action_mean = torch.cat(
            [route_norm, speed_expected], dim=1
        ).detach()  # (B, route_dim+1)

        # --- Residual head features: attention-pooled KV (detached) + base action ---
        # KV tokens carry the full scene context TFv6 attended to; always detached so
        # the actor never backprops into the frozen backbone.
        # Base action (route_norm ++ speed_expected) tells the head what TFv6 already
        # intends to do, allowing it to condition its correction on the base plan.
        spatial_features = (
            self._build_residual_spatial_features()
        )  # (B, residual_spatial_dim)
        base_action = torch.cat(
            [route_norm, speed_expected], dim=1
        ).detach()  # (B, route_dim+1) — detach: no gradient through frozen TFv6 output
        residual_features = torch.cat(
            [spatial_features, base_action], dim=1
        )  # (B, residual_in_dim)

        # --- Residual head: means + log_stds for basis coefficients ---
        # Output layout: [n_0..n_{rank-1}, speed_mean, log_σ_0..log_σ_{rank-1}, log_speed_σ]
        head_out = self.residual_head(residual_features).float()  # (B, 2*(rank+1))
        rank = self.residual_route_rank
        coeff_means = head_out[:, : rank + 1]  # (B, rank+1)
        coeff_log_stds = head_out[:, rank + 1 :].clamp(
            self.log_std_min, self.log_std_max
        )  # (B, rank+1)

        # --- Fixed log_stds when disable_learned_noise_head=True ---
        # Mirrors the normal-path behaviour: log_stds stay at their init values.
        if self.disable_learned_noise_head:
            fixed_route_ls = torch.full(
                (coeff_log_stds.shape[0], rank),
                self._residual_fixed_route_log_std,
                device=coeff_log_stds.device,
                dtype=coeff_log_stds.dtype,
            )
            fixed_speed_ls = torch.full(
                (coeff_log_stds.shape[0], 1),
                self._residual_fixed_speed_log_std,
                device=coeff_log_stds.device,
                dtype=coeff_log_stds.dtype,
            )
            coeff_log_stds = torch.cat([fixed_route_ls, fixed_speed_ls], dim=1)

        route_coeff_means = coeff_means[:, :rank]  # (B, rank)
        speed_coeff_mean = coeff_means[:, rank:]  # (B, 1)

        disable_route = self.disable_residual_route

        basis = self.residual_route_basis.to(
            device=route_coeff_means.device, dtype=route_coeff_means.dtype
        )

        # --- Distribution over coefficients ---
        # When route is disabled the distribution is 1-D (speed only): route coefficients
        # carry no gradient and are never applied, so exclude them from log_prob entirely.
        if disable_route:
            self._residual_dist.proba_distribution(
                speed_coeff_mean, coeff_log_stds[:, rank:]
            )
        else:
            self._residual_dist.proba_distribution(coeff_means, coeff_log_stds)
        dist = self._residual_dist

        # --- Action mean (base + deterministic residual correction) ---
        delta_route_mean = (
            torch.zeros_like(route_norm)
            if disable_route
            else self.residual_alpha * route_coeff_means @ basis.t()
        )  # (B, route_dim)
        delta_speed_mean = self.residual_alpha_speed * speed_coeff_mean  # (B, 1)
        action_mean = torch.cat(
            [route_norm + delta_route_mean, speed_expected + delta_speed_mean], dim=1
        )  # (B, route_dim+1)

        # --- Sample or recover coefficients ---
        if actions is None:
            if sample_type in ("mean", "mode", "deterministic"):
                sampled_coeffs = speed_coeff_mean if disable_route else coeff_means
            else:
                sampled_coeffs = (
                    dist.sample()
                )  # (B, 1) if disable_route else (B, rank+1)

            if disable_route:
                speed_coeff_s = sampled_coeffs  # (B, 1)
                delta_route_s = torch.zeros_like(route_norm)
            else:
                route_coeffs_s = sampled_coeffs[:, :rank]  # (B, rank)
                speed_coeff_s = sampled_coeffs[:, rank:]  # (B, 1)
                delta_route_s = (
                    self.residual_alpha * route_coeffs_s @ basis.t()
                )  # (B, route_dim)

            delta_speed_s = self.residual_alpha_speed * speed_coeff_s  # (B, 1)
            actions = torch.cat(
                [route_norm + delta_route_s, speed_expected + delta_speed_s], dim=1
            )  # (B, route_dim+1) — full action for env.step()
            coeffs_for_log_prob = sampled_coeffs  # (B,1) or (B, rank+1)
        else:
            # Update step: recover basis coefficients from stored action.
            # TFv6 is frozen so route_norm and speed_expected are identical to rollout.
            route_dim = self.action_codec.route_dim
            stored_speed = actions[:, route_dim:]
            speed_coeff_s = (stored_speed - speed_expected) / (
                self.residual_alpha_speed + 1e-8
            )  # (B, 1)
            if disable_route:
                coeffs_for_log_prob = speed_coeff_s
            else:
                stored_route = actions[:, :route_dim]
                route_coeffs_s = (
                    (stored_route - route_norm) / (self.residual_alpha + 1e-8) @ basis
                )  # (B, rank)
                coeffs_for_log_prob = torch.cat(
                    [route_coeffs_s, speed_coeff_s], dim=1
                )  # (B, rank+1)
            # Replace actions with coefficients so the returned tensor has consistent shape.
            actions = coeffs_for_log_prob

        log_prob = dist.log_prob(coeffs_for_log_prob)
        entropy = dist.entropy()
        if entropy.ndim > 1:
            entropy = entropy.sum(1)

        values = self.value_head(self._build_critic_input(obs_dict))

        # Effective per-dim log_std in action space (for spatial std profile in debug viz).
        route_coeff_stds_sq = torch.exp(coeff_log_stds[:, :rank]).pow(2)  # (B, rank)
        basis_sq = basis.pow(2)  # (route_dim, rank)
        route_var_per_dim = (self.residual_alpha**2) * (
            route_coeff_stds_sq @ basis_sq.t()
        )  # (B, route_dim)
        route_log_std_eff = 0.5 * torch.log(route_var_per_dim + 1e-8)  # (B, route_dim)
        speed_log_std_eff = torch.log(
            self.residual_alpha_speed * torch.exp(coeff_log_stds[:, rank:]) + 1e-8
        )  # (B, 1)
        diag_log_std = torch.cat(
            [route_log_std_eff, speed_log_std_eff], dim=1
        )  # (B, route_dim+1)

        return (
            actions,  # (B, rank+1) at update; (B, 21) at rollout
            log_prob,
            entropy,
            values,
            None,  # exp_loss (unused)
            action_mean.detach(),  # (B, 21) base+residual mean for viz
            head_out.detach(),  # (B, 2*(rank+1)) coeff means+log_stds for buffer
            dist,
            target_speed_logits.detach(),  # (B, bins) base TFv6 speed logits
            diag_log_std.detach(),  # (B, 21) effective action-space log_std for viz
            lstm_state,
        )

    @torch.no_grad()
    def get_reference_policy_outputs(
        self, obs_dict
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
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
        ref_action_mean = self.action_codec.encode(
            ref_route, ref_waypoints, ref_target_speed
        ).float()
        ref_target_speed_logits = (
            ref_predictions.pred_target_speed_distribution.float().detach()
            if ref_predictions.pred_target_speed_distribution is not None
            else None
        )
        return ref_action_mean, ref_target_speed_logits

    @torch.no_grad()
    def get_reference_action_mean(self, obs_dict) -> torch.Tensor:
        ref_action_mean, _ = self.get_reference_policy_outputs(obs_dict)
        return ref_action_mean

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
        return self.value_head(self._build_critic_input(obs_dict))

    def forward(
        self,
        obs_dict,
        actions=None,
        sample_type: str = "sample",
        exploration_suggests=None,
        lstm_state=None,
        done=None,
    ) -> tuple:
        if self.use_residual_policy:
            return self._forward_residual(obs_dict, actions, sample_type, lstm_state)

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
        target_speed_logits = (
            predictions.pred_target_speed_distribution.float()
            if predictions.pred_target_speed_distribution is not None
            else None
        )

        action_mean = self.action_codec.encode(route, waypoints, target_speed).float()
        value_features = self._build_value_and_noise_features(obs_dict)
        noise_pred = self.predict_action_noise(value_features)
        dist, noise_diag = self.action_noise_codec.proba_distribution(
            action_mean,
            noise_pred,
            speed_logits=target_speed_logits,
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

        values = self.value_head(self._build_critic_input(obs_dict))

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
            dist,
            None if target_speed_logits is None else target_speed_logits.detach(),
            noise_diag.diag_log_std.detach(),
            lstm_state,
        )
