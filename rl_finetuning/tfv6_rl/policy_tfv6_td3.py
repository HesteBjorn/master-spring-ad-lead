"""TD3 actor and Q-network classes for TFv6 residual RL.

Both actor and Q-networks receive a shared ``TFv6ResidualBackbone`` instance so
TFv6 runs only once per forward pass. The design is:

  - Actor *owns* the backbone (it is a registered ``nn.Module`` child), so
    ``actor.parameters()`` includes ``backbone.residual_cnn`` and
    ``backbone.residual_status_proj``.
  - Q-networks hold a *reference* to the backbone (not a child module) so
    ``qf.parameters()`` only contains ``qf.q_head``. This avoids duplicate
    parameter registration when actor and Q-networks share one backbone.

Usage pattern (training loop):
    # Online networks share one backbone.
    backbone       = TFv6ResidualBackbone(...)
    actor          = TFv6ResidualActorTD3(backbone, config)
    qf1            = TFv6ResidualQNetworkTD3(backbone, config)
    qf2            = TFv6ResidualQNetworkTD3(backbone, config)
    # Target networks share a separate target_backbone (deepcopy of backbone).
    target_backbone = deepcopy(backbone)
    actor_target   = TFv6ResidualActorTD3(target_backbone, config)
    qf1_target     = TFv6ResidualQNetworkTD3(target_backbone, config)
    qf2_target     = TFv6ResidualQNetworkTD3(target_backbone, config)

    # Collect step (no_grad):
    coeff = actor.forward_coeffs(obs)          # sets backbone._last_base_action_mean
    action = actor.coeffs_to_action(coeff)     # combines base + alpha*correction

    # Critic update:
    with no_grad():
        next_coeff = actor_target.forward_coeffs(next_obs)  # sets target_backbone state
        q1_next = qf1_target(next_obs, next_coeff)           # reuses target_backbone state
        q2_next = qf2_target(next_obs, next_coeff)           # reuses target_backbone state
    q1 = qf1(obs, batch_coeff)                  # sets backbone state (via actor's backbone)
    q2 = qf2(obs, batch_coeff)

    # Actor update:
    pred_coeff = actor.forward_coeffs(obs)     # sets backbone._last_base_action_mean
    actor_loss = -qf1.forward_with_cached_backbone(pred_coeff, priv).mean()
"""

from __future__ import annotations

import torch
from torch import nn

from rl_finetuning.tfv6_rl.residual_backbone import TFv6ResidualBackbone


class TFv6ResidualActorTD3(nn.Module):
    """Deterministic residual actor for TD3.

    Outputs a full action in normalized space: [route_dim+1].
    The correction is parameterized in a low-dimensional coefficient space
    of size ``rank+1`` (route basis coefficients + speed coefficient).

    Key methods:
        ``forward_coeffs(obs_dict)``    → [B, rank+1]  raw coefficients
        ``coeffs_to_action(coeffs)``    → [B, route_dim+1]  full action in [-1,1]
        ``forward(obs_dict)``           → [B, route_dim+1]  chains both
    """

    def __init__(self, backbone: TFv6ResidualBackbone, rl_config=None) -> None:
        super().__init__()
        # Backbone is a registered child: actor owns backbone.residual_cnn +
        # backbone.residual_status_proj. TFv6 inside backbone is frozen.
        self.backbone = backbone
        token_dim = backbone.value_token_dim
        rank = backbone.residual_route_rank
        hidden_dim = 512
        # When route correction is disabled, output only the speed scalar.
        # This drops the route coefficient entirely so it cannot introduce noisy
        # gradients into the shared hidden layer during actor/critic updates.
        eff_rank = 0 if backbone.disable_residual_route else rank
        # Output head: [B, 2*token_dim] → [B, eff_rank+1] coefficient means in (-1, 1).
        # Tanh bounds coefficients so alpha*coeff is always within [-alpha, +alpha],
        # matches the [-1,1] clamp assumed by target policy smoothing, and keeps
        # gradients alive (no dead zone from action.clamp saturation).
        # Final linear zero-initialized so actor starts at TFv6 base policy (tanh(0)=0).
        self.residual_out = nn.Sequential(
            nn.Linear(token_dim * 2 + backbone.speed_history_len, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, eff_rank + 1),
            nn.Tanh(),
        )
        nn.init.zeros_(self.residual_out[-2].weight)
        nn.init.zeros_(self.residual_out[-2].bias)

    @property
    def rank(self) -> int:
        return self.backbone.residual_route_rank

    @property
    def effective_rank(self) -> int:
        """Route rank actually used: 0 when route correction is disabled."""
        return 0 if self.backbone.disable_residual_route else self.rank

    @property
    def action_dim(self) -> int:
        return self.backbone.action_codec.action_dim

    def forward_coeffs(
        self, obs_dict: dict, detach_features: bool = False
    ) -> torch.Tensor:
        """Run backbone + output head → raw coefficients [B, rank+1].

        Side-effect: sets ``backbone._last_base_action_mean`` and caches TFv6
        internal state (``bev_features``, ``kv``) for reuse by Q-networks.

        Args:
            detach_features: if True, stop gradients between the CNN encoder and
                the residual head. Use during actor updates so the encoder trains
                from critic loss only (ResFiT / DrQ-v2 pattern).
        """
        _, _, base_action_mean = self.backbone.get_base_action(obs_dict)
        features = self.backbone.get_residual_features(base_action_mean)
        if detach_features:
            features = features.detach()
        if self.backbone.speed_history_len > 0 and "speed_history" in obs_dict:
            features = torch.cat([features, obs_dict["speed_history"].float()], dim=1)
        return self.residual_out(features)

    def forward_coeffs_from_features(
        self,
        feature_obs: dict[str, torch.Tensor],
        detach_features: bool = False,
    ) -> torch.Tensor:
        """Like forward_coeffs() but skips the frozen TFv6 forward pass entirely.

        Loads pre-encoded features from the replay buffer via set_feature_cache(),
        then runs only the trainable residual CNN and output head.

        Args:
            feature_obs: dict with keys bev, kv_status_mean, base_action_mean
                         (and optionally privileged_measurements) as batch tensors.
            detach_features: stop gradients between CNN encoder and output head.
        """
        self.backbone.set_feature_cache(
            feature_obs["bev"],
            feature_obs["kv_status_mean"],
            feature_obs["base_action_mean"],
        )
        features = self.backbone.get_residual_features(
            self.backbone._last_base_action_mean
        )
        if detach_features:
            features = features.detach()
        if self.backbone.speed_history_len > 0 and "speed_history" in feature_obs:
            features = torch.cat(
                [features, feature_obs["speed_history"].float()], dim=1
            )
        return self.residual_out(features)

    def coeffs_to_action(self, coeff_means: torch.Tensor) -> torch.Tensor:
        """Convert coefficient vector [B, rank+1] to full action [B, route_dim+1].

        Applies: action = base_action + alpha * correction.
        Route is clamped to [-1, 1] (lateral corrections are symmetric).
        Speed is clamped to [0, 1]: negative target speed and near-zero target speed
        both trigger full braking in the PID controller, so the entire negative speed
        range is a behaviorally collapsed dead zone — we exclude it.
        Uses the base action cached by the most recent ``forward_coeffs`` call.
        """
        base_action = self.backbone._last_base_action_mean
        if base_action is None:
            raise RuntimeError("Call forward_coeffs() before coeffs_to_action().")
        route_dim = self.backbone.action_codec.route_dim
        eff_rank = self.effective_rank
        basis = self.backbone.residual_route_basis.to(
            device=coeff_means.device, dtype=coeff_means.dtype
        )
        disable_route = self.backbone.disable_residual_route

        route_base = base_action[:, :route_dim]
        speed_base = base_action[:, route_dim:]
        route_coeff = coeff_means[:, :eff_rank]
        speed_coeff = coeff_means[:, eff_rank:]

        if disable_route:
            delta_route = torch.zeros_like(route_base)
        else:
            delta_route = self.backbone.residual_alpha * (route_coeff @ basis.t())

        delta_speed = self.backbone.residual_alpha_speed * speed_coeff
        route_action = (route_base + delta_route).clamp(-1.0, 1.0)
        speed_action = (speed_base + delta_speed).clamp(0.0, 1.0)
        return torch.cat([route_action, speed_action], dim=1)

    def forward(self, obs_dict: dict) -> torch.Tensor:
        """Full actor forward: obs_dict → full action [B, route_dim+1].

        Chains ``forward_coeffs`` + ``coeffs_to_action``. Used during rollout.
        """
        coeff_means = self.forward_coeffs(obs_dict)
        return self.coeffs_to_action(coeff_means)


class TFv6ResidualQNetworkTD3(nn.Module):
    """Q-network for TD3 residual RL.

    Takes coefficient-space actions [B, rank+1] instead of full actions
    [B, route_dim+1]. This keeps the Q-function input low-dimensional (2–3
    dims vs. 21), matching the residual parameterization of the actor.

    The backbone is held as a plain attribute (NOT a registered child module),
    so ``qf.parameters()`` only contains ``qf.q_head`` weights. The backbone
    must be shared with the corresponding actor instance — TFv6 must have
    already been run via ``actor.forward_coeffs()`` before calling this.

    Instantiate two independent instances (qf1, qf2) sharing the same backbone.
    Q-head architecture mirrors the PPO value_head (LayerNorm + MLP).
    """

    def __init__(
        self,
        backbone: TFv6ResidualBackbone,
        rl_config=None,
    ) -> None:
        super().__init__()
        # Plain attribute — NOT registered as child module.
        # backbone.parameters() are owned by the actor, not this Q-network.
        object.__setattr__(self, "_backbone_ref", backbone)

        token_dim = backbone.value_token_dim
        rank = backbone.residual_route_rank
        eff_rank = 0 if backbone.disable_residual_route else rank
        action_dim = eff_rank + 1

        self.use_privileged_measurements: bool = False
        self.num_privileged_measurements: int = 0
        if rl_config is not None:
            self.use_privileged_measurements = bool(
                getattr(rl_config, "use_value_measurements", False)
            )
            if self.use_privileged_measurements:
                from rl_finetuning.tfv6_rl.privileged_measurements import (
                    privileged_measurement_dim,
                )

                self.num_privileged_measurements = privileged_measurement_dim(rl_config)

        priv_dim = (
            self.num_privileged_measurements if self.use_privileged_measurements else 0
        )
        self.speed_history_len: int = backbone.speed_history_len
        q_in_dim = token_dim * 2 + action_dim + priv_dim + self.speed_history_len

        q_hidden_dim = 512
        # Q-head mirrors PPO value_head, but uses a wider TD3 critic head.
        self.q_head = nn.Sequential(
            nn.LayerNorm(q_in_dim),
            nn.Linear(q_in_dim, q_hidden_dim),
            nn.LayerNorm(q_hidden_dim),
            nn.GELU(),
            nn.Linear(q_hidden_dim, q_hidden_dim),
            nn.LayerNorm(q_hidden_dim),
            nn.GELU(),
            nn.Linear(q_hidden_dim, 1),
        )

    @property
    def backbone(self) -> TFv6ResidualBackbone:
        return object.__getattribute__(self, "_backbone_ref")

    def forward(
        self,
        obs_dict: dict,
        action_coeffs: torch.Tensor,
        privileged: torch.Tensor | None = None,
        speed_history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute Q(obs, action_coeffs) reusing cached backbone state.

        ``actor.forward_coeffs(obs_dict)`` (or ``actor_target.forward_coeffs``)
        must have been called first to populate ``backbone._last_base_action_mean``
        and the TFv6 internal cache (``bev_features``, ``kv``).

        Args:
            obs_dict:      observation dict (used only to locate backbone — not re-run)
            action_coeffs: [B, rank+1]  coefficient-space action
            privileged:    [B, num_priv] optional privileged measurements
            speed_history: [B, speed_history_len] optional recent ego speeds

        Returns:
            q_value: [B, 1]
        """
        base_action = self.backbone._last_base_action_mean
        if base_action is None:
            raise RuntimeError(
                "Call actor.forward_coeffs() before Q-network forward()."
            )
        features = self.backbone.get_residual_features(base_action)

        parts = [features, action_coeffs.float()]
        if self.use_privileged_measurements and privileged is not None:
            parts.append(privileged.float())
        if self.speed_history_len > 0 and speed_history is not None:
            parts.append(speed_history.float())
        x = torch.cat(parts, dim=1)
        return self.q_head(x)

    def forward_with_cached_backbone(
        self,
        action_coeffs: torch.Tensor,
        privileged: torch.Tensor | None = None,
        detach_features: bool = False,
        speed_history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Like forward() but without obs_dict, purely from cached backbone state.

        Convenience method for the actor-loss computation where the backbone
        was already run by ``actor.forward_coeffs()`` in the same step.

        Args:
            detach_features: if True, stop gradients from the Q-head back into
                the CNN encoder. Use during actor updates (ResFiT / DrQ-v2 pattern).
            speed_history: [B, speed_history_len] optional recent ego speeds.
        """
        base_action = self.backbone._last_base_action_mean
        if base_action is None:
            raise RuntimeError(
                "Call actor.forward_coeffs() before forward_with_cached_backbone()."
            )
        features = self.backbone.get_residual_features(base_action)
        if detach_features:
            features = features.detach()

        parts = [features, action_coeffs.float()]
        if self.use_privileged_measurements and privileged is not None:
            parts.append(privileged.float())
        if self.speed_history_len > 0 and speed_history is not None:
            parts.append(speed_history.float())
        x = torch.cat(parts, dim=1)
        return self.q_head(x)
