"""Frozen TFv6 + transformer token encoder backbone for residual TD3.

Replaces TFv6ResidualBackbone when --architecture transformer is used.
Instead of CNN + GAP + status mean-pool, the BEV feature map is encoded
via a small convolutional stem into spatial tokens, and all TFv6 status
tokens are preserved individually — giving the decoder full spatial and
semantic resolution to attend over.

Trainable encoder attributes are named bev_token_encoder and
status_token_encoder. The training script accesses them exclusively via
the encoder_parameters() / encoder_modules() / encoder_state_dict() /
load_encoder_state_dict() interface, so names don't need to match
TFv6ResidualBackbone.

Key interface difference from TFv6ResidualBackbone:
  get_residual_features() returns [B, N_tokens, D] instead of [B, 2*D].
  encode_obs_to_features() / feature_obs_space() use kv_status_tokens
  [N_status, D_tfv6] instead of kv_status_mean [D_tfv6].
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
from torch import nn

from lead.inference.config_closed_loop import ClosedLoopConfig
from lead.tfv6.tfv6 import TFv6
from lead.training.config_training import TrainingConfig
from rl_finetuning.tfv6_rl.action_codec import ActionCodec, infer_action_head_usage
from rl_finetuning.tfv6_rl.action_noise_codec import build_route_basis


def _load_training_config(checkpoint_dir: str) -> TrainingConfig:
    config_path = os.path.join(checkpoint_dir, "config.json")
    with open(config_path, encoding="utf-8") as f:
        loaded = json.load(f)
    return TrainingConfig(loaded, raise_error_on_missing_key=False)


def _find_model_file(checkpoint_dir: str, prefix: str = "model") -> str:
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


# ── Token encoder modules ─────────────────────────────────────────────────────


class BEVTokenEncoder(nn.Module):
    """Conv stem + learned positional embedding → BEV spatial tokens.

    Freshly initialized — does not reuse TFv6's dimension_adapter or
    positional embeddings. Architecture mirrors TFv6's context encoder:
    k=1 channel adapter then k=3 spatial conv, followed by a learned 2D
    positional embedding (TFv6 uses status_pos_embedding in the same way
    for status tokens; we use the same pattern for spatial tokens here).
    """

    def __init__(self, in_channels: int, d_model: int, n_h: int, n_w: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.pos_embedding = nn.Parameter(torch.zeros(1, n_h * n_w, d_model))
        nn.init.uniform_(self.pos_embedding)

    def forward(self, bev: torch.Tensor) -> torch.Tensor:
        # bev: [B, C, H, W]
        x = self.conv(bev)  # [B, D, H, W]
        x = x.flatten(2).transpose(1, 2)  # [B, H*W, D]
        return x + self.pos_embedding  # [B, N_spatial, D]


class StatusTokenEncoder(nn.Module):
    """Project TFv6 status tokens and encode base action as two separate tokens.

    Mirrors TFv6's PlanningContextEncoder status path: linear projection of
    each status token + learned positional embedding (same init as TFv6's
    status_pos_embedding). The base action is split into a route token and
    a speed token so the decoder can attend to them independently.
    """

    def __init__(
        self,
        status_dim: int,
        d_model: int,
        n_status: int,
        route_dim: int,
    ):
        super().__init__()
        self.proj = nn.Linear(status_dim, d_model)
        self.status_pos_embedding = nn.Parameter(torch.zeros(1, n_status, d_model))
        nn.init.uniform_(self.status_pos_embedding)
        self.route_encoder = nn.Linear(route_dim, d_model)
        self.speed_encoder = nn.Linear(1, d_model)

    def forward(
        self,
        status_tokens: torch.Tensor,  # [B, N_status, status_dim]
        base_route: torch.Tensor,  # [B, route_dim]
        base_speed: torch.Tensor,  # [B, 1]
    ) -> torch.Tensor:  # [B, N_status + 2, d_model]
        status = (
            self.proj(status_tokens) + self.status_pos_embedding
        )  # [B, N_status, D]
        route_tok = self.route_encoder(base_route).unsqueeze(1)  # [B, 1, D]
        speed_tok = self.speed_encoder(base_speed).unsqueeze(1)  # [B, 1, D]
        return torch.cat([status, route_tok, speed_tok], dim=1)  # [B, N_status+2, D]


# ── Backbone ──────────────────────────────────────────────────────────────────


class TFv6TransformerBackbone(nn.Module):
    """Frozen TFv6 base policy + transformer token encoder.

    The trainable modules are:
      bev_token_encoder    = BEVTokenEncoder    (BEV grid → spatial tokens)
      status_token_encoder = StatusTokenEncoder (status + base action → tokens)
    """

    def __init__(
        self,
        tfv6_checkpoint: str,
        tfv6_prefix: str = "model",
        device: torch.device | None = None,
        rl_config=None,
        d_model: int = 128,
    ) -> None:
        super().__init__()

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.d_model = d_model

        # ── TFv6 (frozen) ─────────────────────────────────────────────────────
        if not tfv6_checkpoint:
            raise ValueError("tfv6_checkpoint must be provided")
        self.training_config = _load_training_config(tfv6_checkpoint)
        self.tfv6 = TFv6(self.device, self.training_config)
        weights_path = _find_model_file(tfv6_checkpoint, prefix=tfv6_prefix)
        state_dict = torch.load(
            weights_path, map_location=self.device, weights_only=True
        )
        self.tfv6.load_state_dict(state_dict, strict=True)
        self.tfv6.to(self.device)
        for param in self.tfv6.parameters():
            param.requires_grad_(False)
        self.tfv6.eval()

        self.autocast_dtype = self.training_config.torch_float_type
        self.autocast_enabled = (
            self.training_config.use_mixed_precision_training
            and self.device.type == "cuda"
        )

        # ── Action codec ──────────────────────────────────────────────────────
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
        if not self.action_codec.predict_route:
            raise ValueError("TFv6TransformerBackbone requires route predictions.")
        if not self.action_codec.predict_target_speed:
            raise ValueError(
                "TFv6TransformerBackbone requires target speed predictions."
            )

        # ── Hyperparameters from rl_config ────────────────────────────────────
        self.skip_perception_heads: bool = True
        self.speed_temperature: float = 1.0
        self.residual_route_rank: int = 2
        self.residual_alpha: float = 0.15
        self.residual_alpha_speed: float = 0.15
        self.disable_residual_route: bool = False
        if rl_config is not None:
            self.skip_perception_heads = bool(
                getattr(rl_config, "skip_perception_heads", self.skip_perception_heads)
            )
            self.speed_temperature = float(
                getattr(rl_config, "speed_temperature", self.speed_temperature)
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
        self.speed_history_len: int = (
            int(getattr(rl_config, "speed_history_len", 0))
            if rl_config is not None
            else 0
        )

        # ── Buffers ───────────────────────────────────────────────────────────
        self.value_token_dim: int = self.training_config.transfuser_token_dim  # 256

        route_basis = build_route_basis(
            num_route_points=self.action_codec.num_route_points,
            route_dim=self.action_codec.route_dim,
            rank=self.residual_route_rank,
        )
        self.register_buffer("residual_route_basis", route_basis, persistent=False)

        speed_bins = torch.tensor(
            self.training_config.target_speed_classes, dtype=torch.float32
        ) / float(self.training_config.max_speed)
        self.register_buffer("residual_speed_bins", speed_bins, persistent=False)

        # ── BEV geometry ──────────────────────────────────────────────────────
        bev_channels = self.tfv6.planning_decoder.planning_context_encoder.dimension_adapter.in_channels
        self._bev_channels: int = bev_channels
        self._bev_h: int = getattr(self.training_config, "lidar_vert_anchors", 10)
        self._bev_w: int = getattr(self.training_config, "lidar_horz_anchors", 12)
        self.n_spatial_tokens: int = self._bev_h * self._bev_w

        n_status = self.tfv6.planning_decoder.planning_context_encoder.status_pos_embedding.shape[
            1
        ]
        self._n_status_tokens: int = n_status

        # ── Trainable encoder modules ─────────────────────────────────────────
        self.bev_token_encoder = BEVTokenEncoder(
            in_channels=bev_channels,
            d_model=d_model,
            n_h=self._bev_h,
            n_w=self._bev_w,
        )
        self.status_token_encoder = StatusTokenEncoder(
            status_dim=self.value_token_dim,
            d_model=d_model,
            n_status=n_status,
            route_dim=self.action_codec.route_dim,
        )

        # ── Internal state ────────────────────────────────────────────────────
        self._last_base_action_mean: torch.Tensor | None = None
        self._last_target_speed_logits: torch.Tensor | None = None
        self._cached_kv_status_tokens: torch.Tensor | None = None
        self.rl_config = rl_config

    # ── Encoder interface ─────────────────────────────────────────────────────

    def encoder_parameters(self) -> list:
        """Trainable backbone parameters for the critic optimizer."""
        return list(self.bev_token_encoder.parameters()) + list(
            self.status_token_encoder.parameters()
        )

    def encoder_modules(self) -> list[nn.Module]:
        """Ordered list of trainable encoder modules for Polyak updates."""
        return [self.bev_token_encoder, self.status_token_encoder]

    def encoder_state_dict(self) -> dict:
        return {
            "bev_token_encoder": self.bev_token_encoder.state_dict(),
            "status_token_encoder": self.status_token_encoder.state_dict(),
        }

    def load_encoder_state_dict(self, state_dict: dict) -> None:
        self.bev_token_encoder.load_state_dict(state_dict["bev_token_encoder"])
        self.status_token_encoder.load_state_dict(state_dict["status_token_encoder"])

    def _extract_tfv6_obs(self, obs_dict: dict) -> dict:
        _skip = {"privileged_measurements", "speed_history"}
        return {k: v for k, v in obs_dict.items() if k not in _skip}

    def get_base_action(
        self, obs_dict: dict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run frozen TFv6, return (route_norm, speed_expected, base_action_mean)."""
        self._cached_kv_status_tokens = None
        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=self.autocast_dtype,
            enabled=self.autocast_enabled,
        ):
            predictions = self.tfv6(
                self._extract_tfv6_obs(obs_dict),
                skip_perception_heads=self.skip_perception_heads,
            )

        route = predictions.pred_route
        assert route is not None
        route = route.float()
        B = route.shape[0]
        route_scale = self.action_codec.route_scale.to(
            device=route.device, dtype=route.dtype
        )
        route_norm = (route / route_scale).reshape(B, -1)

        assert predictions.pred_target_speed_distribution is not None
        target_speed_logits = predictions.pred_target_speed_distribution.float()
        self._last_target_speed_logits = target_speed_logits.detach()
        speed_probs = torch.softmax(
            target_speed_logits / self.speed_temperature, dim=-1
        )
        speed_bins = self.residual_speed_bins.to(
            device=speed_probs.device, dtype=speed_probs.dtype
        )
        speed_expected = (speed_probs * speed_bins).sum(dim=-1, keepdim=True)

        base_action_mean = torch.cat([route_norm, speed_expected], dim=1).detach()
        self._last_base_action_mean = base_action_mean
        return route_norm, speed_expected, base_action_mean

    def get_residual_features(self, base_action_mean: torch.Tensor) -> torch.Tensor:
        """Build KV token sequence from BEV + status + base action.

        Returns:
            [B, N_spatial + N_status + 2, d_model]
        """
        bev = getattr(self.tfv6, "bev_features", None)
        if bev is None:
            raise RuntimeError(
                "bev_features not found — call get_base_action() or set_feature_cache() first."
            )
        bev_tokens = self.bev_token_encoder(bev.detach().float())  # [B, N_spatial, D]

        if self._cached_kv_status_tokens is not None:
            kv_status = self._cached_kv_status_tokens.float()
        else:
            kv = getattr(self.tfv6.planning_decoder, "kv", None)
            if kv is None:
                raise RuntimeError(
                    "Planning decoder kv not found — call get_base_action() first."
                )
            kv_status = kv.detach().float()[
                :, self.n_spatial_tokens :
            ]  # [B, N_status, 256]

        route_dim = self.action_codec.route_dim
        base_route = base_action_mean[:, :route_dim].float()
        base_speed = base_action_mean[:, route_dim:].float()
        context_tokens = self.status_token_encoder(
            kv_status, base_route, base_speed
        )  # [B, N_status+2, D]

        return torch.cat([bev_tokens, context_tokens], dim=1)  # [B, N_total, D]

    def set_feature_cache(
        self,
        bev: torch.Tensor,
        kv_status_tokens: torch.Tensor,
        base_action_mean: torch.Tensor,
    ) -> None:
        """Restore backbone state from replay buffer features."""
        self.tfv6.bev_features = bev
        self._cached_kv_status_tokens = kv_status_tokens
        self._last_base_action_mean = base_action_mean

    def encode_obs_to_features(
        self, obs_dict: dict | None = None, env_idx: int = 0
    ) -> dict[str, np.ndarray]:
        """Extract frozen TFv6 outputs as numpy arrays for replay buffer storage.

        Returns dict with keys:
          bev              [C, H, W]
          kv_status_tokens [N_status, D_tfv6]
          base_action_mean [route_dim + 1]
        """
        if obs_dict is not None:
            with torch.no_grad():
                self.get_base_action(obs_dict)
        bev = self.tfv6.bev_features.detach().float().cpu().numpy()[env_idx]
        kv = getattr(self.tfv6.planning_decoder, "kv", None)
        if kv is None:
            raise RuntimeError("Planning decoder kv not found.")
        kv_status_tokens = (
            kv[env_idx, self.n_spatial_tokens :].float().cpu().numpy()
        )  # [N_status, D_tfv6]
        base_action_mean = self._last_base_action_mean[env_idx].float().cpu().numpy()
        result = {
            "bev": bev,
            "kv_status_tokens": kv_status_tokens,
            "base_action_mean": base_action_mean,
        }
        if self.rl_config and getattr(self.rl_config, "use_value_measurements", False):
            # Privileged measurements are observation-level, not backbone-level.
            # The training loop injects them separately; nothing to do here.
            pass
        return result

    def feature_obs_space(self):
        """Gymnasium Dict space for pre-encoded frozen TFv6 features."""
        from gymnasium import spaces

        obs = {
            "bev": spaces.Box(
                -np.inf,
                np.inf,
                shape=(self._bev_channels, self._bev_h, self._bev_w),
                dtype=np.float32,
            ),
            "kv_status_tokens": spaces.Box(
                -np.inf,
                np.inf,
                shape=(self._n_status_tokens, self.value_token_dim),
                dtype=np.float32,
            ),
            "base_action_mean": spaces.Box(
                -np.inf,
                np.inf,
                shape=(self.action_codec.route_dim + 1,),
                dtype=np.float32,
            ),
        }
        if self.rl_config and getattr(self.rl_config, "use_value_measurements", False):
            from rl_finetuning.tfv6_rl.privileged_measurements import (
                privileged_measurement_dim,
            )

            priv_dim = privileged_measurement_dim(self.rl_config)
            obs["privileged_measurements"] = spaces.Box(
                -np.inf, np.inf, shape=(priv_dim,), dtype=np.float32
            )
        if self.speed_history_len > 0:
            obs["speed_history"] = spaces.Box(
                -np.inf, np.inf, shape=(self.speed_history_len,), dtype=np.float32
            )
        return spaces.Dict(obs)
