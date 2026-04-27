"""Shared frozen TFv6 + residual CNN feature extractor.

Used as the backbone for TFv6ResidualActorTD3 and TFv6ResidualQNetworkTD3.
Encapsulates:
  - Loading and freezing the TFv6 base policy
  - Spatial residual CNN over BEV features
  - Status branch over KV tokens + base action
  - get_base_action(): TFv6 forward → route_norm + E[categorical speed]
  - get_residual_features(): CNN features for residual head / Q-network

Identical logic to the residual path in TFv6PPOPolicy, extracted so that both
the TD3 actor and the twin Q-networks can share a single TFv6 forward pass.
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


class TFv6ResidualBackbone(nn.Module):
    """Frozen TFv6 base policy + residual CNN feature extractor.

    Shared between the TD3 actor and Q-networks so TFv6 runs only once per
    forward pass. The actor calls ``get_base_action`` first (which runs TFv6
    and caches ``bev_features`` + ``kv`` on the TFv6 module); Q-networks then
    call ``get_residual_features`` which reuses that cached state.

    Trainable parameters: ``residual_cnn`` and ``residual_status_proj``.
    TFv6 is always fully frozen.
    """

    def __init__(
        self,
        tfv6_checkpoint: str,
        tfv6_prefix: str = "model",
        device: torch.device | None = None,
        rl_config=None,
    ) -> None:
        super().__init__()

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # ── TFv6 (frozen) ─────────────────────────────────────────────────
        if not tfv6_checkpoint:
            raise ValueError("tfv6_checkpoint must be provided to TFv6ResidualBackbone")
        self.training_config = _load_training_config(tfv6_checkpoint)
        self.tfv6 = TFv6(self.device, self.training_config)
        weights_path = _find_model_file(tfv6_checkpoint, prefix=tfv6_prefix)
        state_dict = torch.load(
            weights_path, map_location=self.device, weights_only=True
        )
        self.tfv6.load_state_dict(state_dict, strict=True)
        self.tfv6.to(self.device)
        # TFv6 is always fully frozen in the residual backbone.
        for param in self.tfv6.parameters():
            param.requires_grad_(False)
        self.tfv6.eval()

        # Autocast settings mirror TFv6 inference behavior.
        self.autocast_dtype = self.training_config.torch_float_type
        self.autocast_enabled = (
            self.training_config.use_mixed_precision_training
            and self.device.type == "cuda"
        )

        # ── Action codec ──────────────────────────────────────────────────
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
            raise ValueError("TFv6ResidualBackbone requires route predictions.")
        if not self.action_codec.predict_target_speed:
            raise ValueError("TFv6ResidualBackbone requires target speed predictions.")

        # ── Hyperparameters from rl_config ────────────────────────────────
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

        # ── Buffers (non-persistent: computed at init, not saved in state_dict) ─
        self.value_token_dim: int = self.training_config.transfuser_token_dim

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

        # ── Residual CNN architecture ──────────────────────────────────────
        token_dim = self.value_token_dim
        bev_channels = self.tfv6.planning_decoder.planning_context_encoder.dimension_adapter.in_channels
        _bev_hidden = 64
        # Spatial branch: 1×1 channel reduction → two 3×3 spatial convs → GAP.
        self.residual_cnn = nn.Sequential(
            nn.Conv2d(bev_channels, _bev_hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(_bev_hidden, _bev_hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(_bev_hidden, token_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        # Status branch: mean-pooled status KV tokens + TFv6 base action → linear.
        residual_action_dim = self.action_codec.route_dim + 1
        self.residual_status_proj = nn.Linear(
            token_dim + residual_action_dim, token_dim
        )

        # Cache set by get_base_action(), read by get_residual_features().
        # Not a Parameter — not saved in state_dict.
        self._last_base_action_mean: torch.Tensor | None = None
        # Stored for debug visualization (target_speed_logits from TFv6).
        self._last_target_speed_logits: torch.Tensor | None = None
        # Pre-encoded kv_status_mean from replay buffer (set by set_feature_cache).
        # None means live path: get_residual_features reads directly from TFv6 planning_decoder.kv.
        self._cached_kv_status_mean: torch.Tensor | None = None
        # BEV grid dimensions and token split (for encode_obs_to_features / feature_obs_space).
        self._bev_channels: int = bev_channels
        self._bev_h: int = getattr(self.training_config, "lidar_vert_anchors", 10)
        self._bev_w: int = getattr(self.training_config, "lidar_horz_anchors", 12)
        self.n_spatial_tokens: int = self._bev_h * self._bev_w
        self.rl_config = rl_config

    def _extract_tfv6_obs(self, obs_dict: dict) -> dict:
        """Strip privileged measurements — not a TFv6 input."""
        return {k: v for k, v in obs_dict.items() if k != "privileged_measurements"}

    def get_base_action(
        self, obs_dict: dict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run frozen TFv6 and compute base route + expected speed.

        Sets ``self._last_base_action_mean`` and caches ``bev_features`` / ``kv``
        on ``self.tfv6`` for reuse by ``get_residual_features``.

        Returns:
            route_norm:       [B, route_dim]    normalized route (no clamp)
            speed_expected:   [B, 1]            E[categorical speed], normalized
            base_action_mean: [B, route_dim+1]  concatenation (detached)
        """
        self._cached_kv_status_mean = None  # live TFv6 run — invalidate buffer cache
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
        assert route is not None, "TFv6ResidualBackbone requires pred_route from TFv6."
        route = route.float()
        B = route.shape[0]
        route_scale = self.action_codec.route_scale.to(
            device=route.device, dtype=route.dtype
        )
        route_norm = (route / route_scale).reshape(B, -1)

        assert predictions.pred_target_speed_distribution is not None, (
            "TFv6ResidualBackbone requires pred_target_speed_distribution from TFv6."
        )
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
        """Build residual CNN features from frozen BEV + status branch.

        Reuses ``bev_features`` and ``kv`` already cached on ``self.tfv6`` from
        the most recent ``get_base_action()`` call. No redundant TFv6 forward.

        Logic is identical to ``TFv6PPOPolicy._build_residual_features()``.

        Args:
            base_action_mean: [B, route_dim+1]  TFv6 base action (route_norm + speed).

        Returns:
            Tensor [B, 2*token_dim]
        """
        bev = getattr(self.tfv6, "bev_features", None)
        if bev is None:
            raise RuntimeError(
                "bev_features not found on tfv6 — call get_base_action() or set_feature_cache() first."
            )

        # Spatial branch: frozen BEV feature map.
        bev = bev.detach().float()  # (B, C, H, W)
        spatial_ctx = self.residual_cnn(bev)  # (B, token_dim)

        # Status branch: pre-encoded from buffer, or live from planning decoder kv.
        if self._cached_kv_status_mean is not None:
            kv_status = self._cached_kv_status_mean
        else:
            kv = getattr(self.tfv6.planning_decoder, "kv", None)
            if kv is None:
                raise RuntimeError(
                    "Planning decoder context tokens (kv) not found — call get_base_action() first."
                )
            kv_status = (
                kv.detach().float()[:, self.n_spatial_tokens :].mean(dim=1)
            )  # (B, token_dim)
        status_ctx = self.residual_status_proj(
            torch.cat([kv_status, base_action_mean.float()], dim=1)
        )  # (B, token_dim)

        return torch.cat([spatial_ctx, status_ctx], dim=1).float()  # (B, 2*token_dim)

    def set_feature_cache(
        self,
        bev: torch.Tensor,
        kv_status_mean: torch.Tensor,
        base_action_mean: torch.Tensor,
    ) -> None:
        """Load pre-encoded frozen features so get_residual_features() skips TFv6.

        Called during training updates to restore backbone state from replay buffer
        features instead of re-running the frozen TFv6 forward pass.
        """
        self.tfv6.bev_features = bev
        self._cached_kv_status_mean = kv_status_mean
        self._last_base_action_mean = base_action_mean

    def encode_obs_to_features(
        self, obs_dict: dict | None = None
    ) -> dict[str, np.ndarray]:
        """Extract frozen TFv6 outputs as numpy arrays for replay buffer storage.

        If obs_dict is provided, runs TFv6 on it first (use for next_obs encoding).
        If None, reads from the cache left by the most recent get_base_action() call.

        Returns dict with keys: bev [C,H,W], kv_status_mean [token_dim], base_action_mean [route_dim+1].
        """
        if obs_dict is not None:
            with torch.no_grad():
                self.get_base_action(obs_dict)
        bev = self.tfv6.bev_features.detach().float().cpu().numpy()[0]  # (C, H, W)
        kv = getattr(self.tfv6.planning_decoder, "kv", None)
        if kv is None:
            raise RuntimeError(
                "Planning decoder kv not found — call get_base_action() first."
            )
        kv_status_mean = (
            kv[0, self.n_spatial_tokens :].float().mean(0).cpu().numpy()
        )  # (token_dim,)
        base_action_mean = (
            self._last_base_action_mean[0].float().cpu().numpy()
        )  # (route_dim+1,)
        return {
            "bev": bev,
            "kv_status_mean": kv_status_mean,
            "base_action_mean": base_action_mean,
        }

    def feature_obs_space(self):
        """Gymnasium Dict space for pre-encoded frozen TFv6 features.

        Used to initialise DictReplayBuffer instead of the raw sensor obs space,
        reducing per-transition size from ~3.5 MB to ~32 KB.
        """
        from gymnasium import spaces

        obs = {
            "bev": spaces.Box(
                -np.inf,
                np.inf,
                shape=(self._bev_channels, self._bev_h, self._bev_w),
                dtype=np.float32,
            ),
            "kv_status_mean": spaces.Box(
                -np.inf, np.inf, shape=(self.value_token_dim,), dtype=np.float32
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
        return spaces.Dict(obs)
