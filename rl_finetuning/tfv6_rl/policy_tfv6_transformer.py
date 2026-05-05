"""Transformer-based TD3 actor and Q-network for TFv6 residual RL.

Architecture (per design):
  Shared BEV token encoder in the backbone (updated by critic_lr only).

  Actor:  ActorDecoder  — 4 queries (1 output + 3 reasoning), 2-layer
          TransformerDecoder at D=128, H=4, FFN=512. Output query 0
          → Linear(D → coeff_dim) → Tanh.

  Critic: CriticDecoder — identical decoder structure. Output query 0
          concatenated with action_coeffs → LayerNorm + MLP → scalar.

All decoder layers use full self-attention + cross-attention with GELU,
matching TFv6's TransformerDecoderLayer configuration. The 3 reasoning
queries enrich the self-attention signal before the primary query is read.

External interface is identical to policy_tfv6_td3.py:
  actor.residual_out       → actor_optimizer
  backbone.bev_token_encoder    → critic_optimizer
  backbone.status_token_encoder → critic_optimizer
  qf.q_head                → critic_optimizer
"""

from __future__ import annotations

import torch
from torch import nn

from rl_finetuning.tfv6_rl.transformer_backbone import TFv6TransformerBackbone

# ── Transformer decoder hyperparameters ──────────────────────────────────────

D_MODEL = 128
N_HEADS = 4
N_LAYERS = 2
N_QUERIES = 4  # 1 output query + 3 free reasoning queries
FFN_DIM = 512  # 4 × D_MODEL, matching TFv6's FFN expansion ratio

# ── Shared decoder base ───────────────────────────────────────────────────────


class _ResidualTransformerDecoder(nn.Module):
    """Learned queries + TransformerDecoder, mirroring TFv6's PlanningDecoder.

    Queries initialized with nn.init.uniform_ (same as TFv6's query parameter).
    TransformerDecoderLayer uses GELU activation, batch_first=True, and a
    final LayerNorm — exactly matching TFv6's transformer_decoder construction.
    dropout=0.0: RL training relies on target networks and exploration noise
    for regularization, not internal dropout.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        n_queries: int,
        ffn_dim: int,
    ):
        super().__init__()
        self.queries = nn.Parameter(torch.zeros(1, n_queries, d_model))
        nn.init.uniform_(self.queries)
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=ffn_dim,
                dropout=0.0,
                activation=nn.GELU(),
                batch_first=True,
            ),
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
        )

    def _decode(self, kv_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            kv_tokens: [B, N_kv, d_model]
        Returns:
            [B, n_queries, d_model]
        """
        B = kv_tokens.shape[0]
        queries = self.queries.repeat(B, 1, 1)
        return self.transformer_decoder(queries, kv_tokens)


# ── Actor decoder ─────────────────────────────────────────────────────────────


class ActorDecoder(_ResidualTransformerDecoder):
    """Actor head: query 0 → Linear → Tanh → [B, coeff_dim].

    The output Linear is zero-initialized so the actor starts at the base
    policy (Tanh(0) = 0 means zero residual correction), matching the
    initialization of residual_out in the CNN actor.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        n_queries: int,
        ffn_dim: int,
        coeff_dim: int,
    ):
        super().__init__(d_model, n_heads, n_layers, n_queries, ffn_dim)
        self.output_head = nn.Linear(d_model, coeff_dim)
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

    def forward(self, kv_tokens: torch.Tensor) -> torch.Tensor:
        decoded = self._decode(kv_tokens)  # [B, N_queries, D]
        return torch.tanh(self.output_head(decoded[:, 0]))  # [B, coeff_dim]


# ── Critic decoder ────────────────────────────────────────────────────────────


class CriticDecoder(_ResidualTransformerDecoder):
    """Critic head: query 0 + action_coeffs → LayerNorm + MLP → [B, 1].

    The action (coeff_dim) and optional privileged / speed-history inputs
    are concatenated with the decoded primary query before the Q head.
    This matches the information structure of the CNN critic while letting
    the transformer attend over the full KV context before combining with
    the action.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        n_queries: int,
        ffn_dim: int,
        action_dim: int,
    ):
        super().__init__(d_model, n_heads, n_layers, n_queries, ffn_dim)
        q_in_dim = d_model + action_dim
        self.q_head = nn.Sequential(
            nn.LayerNorm(q_in_dim),
            nn.Linear(q_in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self, kv_tokens: torch.Tensor, action_input: torch.Tensor
    ) -> torch.Tensor:
        decoded = self._decode(kv_tokens)  # [B, N_queries, D]
        x = torch.cat(
            [decoded[:, 0], action_input.float()], dim=-1
        )  # [B, D + action_dim]
        return self.q_head(x)  # [B, 1]


# ── Actor ─────────────────────────────────────────────────────────────────────


class TFv6TransformerActorTD3(nn.Module):
    """Transformer residual actor for TD3.

    self.backbone     = TFv6TransformerBackbone (plain reference, NOT a child module).
    self.residual_out = ActorDecoder            (used by actor_optimizer).

    Interface is identical to TFv6ResidualActorTD3: same method names and
    signatures, same coeffs_to_action logic.
    """

    def __init__(self, backbone: TFv6TransformerBackbone, rl_config=None) -> None:
        super().__init__()
        object.__setattr__(self, "_backbone_ref", backbone)
        eff_rank = (
            0 if backbone.disable_residual_route else backbone.residual_route_rank
        )
        self.residual_out = ActorDecoder(
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            n_queries=N_QUERIES,
            ffn_dim=FFN_DIM,
            coeff_dim=eff_rank + 1,
        )

    @property
    def backbone(self) -> TFv6TransformerBackbone:
        return object.__getattribute__(self, "_backbone_ref")

    @property
    def rank(self) -> int:
        return self.backbone.residual_route_rank

    @property
    def effective_rank(self) -> int:
        return 0 if self.backbone.disable_residual_route else self.rank

    @property
    def action_dim(self) -> int:
        return self.backbone.action_codec.action_dim

    def forward_coeffs(
        self, obs_dict: dict, detach_features: bool = False
    ) -> torch.Tensor:
        """Run backbone + actor decoder → coefficients [B, eff_rank+1].

        Side-effect: sets backbone._last_base_action_mean and caches TFv6
        internal state for reuse by Q-networks.

        Args:
            detach_features: stop gradients from KV tokens into the actor
                decoder. The encoder trains from critic loss only (same
                ResFiT / DrQ-v2 pattern as the CNN variant).
        """
        _, _, base_action_mean = self.backbone.get_base_action(obs_dict)
        kv_tokens = self.backbone.get_residual_features(base_action_mean)
        if detach_features:
            kv_tokens = kv_tokens.detach()
        return self._forward_from_kv(kv_tokens, obs_dict.get("speed_history"))

    def _forward_from_kv(
        self,
        kv_tokens: torch.Tensor,
        speed_history: torch.Tensor | None,
    ) -> torch.Tensor:
        # speed_history is not used in the transformer actor — the transformer
        # can already attend to the base_speed token in the KV sequence.
        # It is accepted here only for interface compatibility.
        return self.residual_out(kv_tokens)

    def forward_coeffs_from_features(
        self,
        feature_obs: dict[str, torch.Tensor],
        detach_features: bool = False,
    ) -> torch.Tensor:
        """Like forward_coeffs() but skips frozen TFv6 using replay buffer features."""
        self.backbone.set_feature_cache(
            feature_obs["bev"],
            feature_obs["kv_status_tokens"],
            feature_obs["base_action_mean"],
        )
        kv_tokens = self.backbone.get_residual_features(
            self.backbone._last_base_action_mean
        )
        if detach_features:
            kv_tokens = kv_tokens.detach()
        return self._forward_from_kv(kv_tokens, feature_obs.get("speed_history"))

    def coeffs_to_action(self, coeff_means: torch.Tensor) -> torch.Tensor:
        """Convert coefficients [B, eff_rank+1] → full action [B, route_dim+1]."""
        base_action = self.backbone._last_base_action_mean
        if base_action is None:
            raise RuntimeError("Call forward_coeffs() before coeffs_to_action().")
        route_dim = self.backbone.action_codec.route_dim
        eff_rank = self.effective_rank
        basis = self.backbone.residual_route_basis.to(
            device=coeff_means.device, dtype=coeff_means.dtype
        )

        route_base = base_action[:, :route_dim]
        speed_base = base_action[:, route_dim:]
        route_coeff = coeff_means[:, :eff_rank]
        speed_coeff = coeff_means[:, eff_rank:]

        delta_route = (
            torch.zeros_like(route_base)
            if self.backbone.disable_residual_route
            else self.backbone.residual_alpha * (route_coeff @ basis.t())
        )
        delta_speed = self.backbone.residual_alpha_speed * speed_coeff
        route_action = (route_base + delta_route).clamp(-1.0, 1.0)
        speed_action = (speed_base + delta_speed).clamp(0.0, 1.0)
        return torch.cat([route_action, speed_action], dim=1)

    def forward(self, obs_dict: dict) -> torch.Tensor:
        coeff_means = self.forward_coeffs(obs_dict)
        return self.coeffs_to_action(coeff_means)


# ── Q-network ─────────────────────────────────────────────────────────────────


class TFv6TransformerQNetworkTD3(nn.Module):
    """Transformer Q-network for TD3.

    self.backbone = TFv6TransformerBackbone (registered child, owned by critic, TFv6 is frozen).
    self.q_head   = CriticDecoder (used by critic_optimizer).

    Interface is identical to TFv6ResidualQNetworkTD3.
    """

    def __init__(
        self,
        backbone: TFv6TransformerBackbone,
        rl_config=None,
    ) -> None:
        super().__init__()
        self.backbone = backbone

        eff_rank = (
            0 if backbone.disable_residual_route else backbone.residual_route_rank
        )
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

        self.speed_history_len: int = backbone.speed_history_len

        extra_dim = (
            self.num_privileged_measurements if self.use_privileged_measurements else 0
        ) + self.speed_history_len
        self.q_head = CriticDecoder(
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            n_queries=N_QUERIES,
            ffn_dim=FFN_DIM,
            action_dim=action_dim + extra_dim,
        )

    def forward(
        self,
        obs_dict: dict,
        action_coeffs: torch.Tensor,
        privileged: torch.Tensor | None = None,
        speed_history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        base_action = self.backbone._last_base_action_mean
        if base_action is None:
            raise RuntimeError(
                "Call actor.forward_coeffs() before Q-network forward()."
            )
        kv_tokens = self.backbone.get_residual_features(base_action)
        return self._q_forward(kv_tokens, action_coeffs, privileged, speed_history)

    def forward_with_cached_backbone(
        self,
        action_coeffs: torch.Tensor,
        privileged: torch.Tensor | None = None,
        detach_features: bool = False,
        speed_history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        base_action = self.backbone._last_base_action_mean
        if base_action is None:
            raise RuntimeError(
                "Call actor.forward_coeffs() before forward_with_cached_backbone()."
            )
        kv_tokens = self.backbone.get_residual_features(base_action)
        if detach_features:
            kv_tokens = kv_tokens.detach()
        return self._q_forward(kv_tokens, action_coeffs, privileged, speed_history)

    def _q_forward(
        self,
        kv_tokens: torch.Tensor,
        action_coeffs: torch.Tensor,
        privileged: torch.Tensor | None,
        speed_history: torch.Tensor | None,
    ) -> torch.Tensor:
        action_input = action_coeffs.float()
        if self.use_privileged_measurements and privileged is not None:
            action_input = torch.cat([action_input, privileged.float()], dim=-1)
        if self.speed_history_len > 0 and speed_history is not None:
            action_input = torch.cat([action_input, speed_history.float()], dim=-1)
        return self.q_head(kv_tokens, action_input)
