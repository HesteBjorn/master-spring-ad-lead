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
  actor.residual_out            → actor_optimizer
  backbone.bev_token_encoder    → critic_optimizer
  backbone.status_token_encoder → critic_optimizer
  qf.q_head                     → critic_optimizer
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


# ── Action token encoder ──────────────────────────────────────────────────────


class CriticActionTokenEncoder(nn.Module):
    """Encode residual action coefficients as separate KV tokens.

    Mirrors StatusTokenEncoder: separate Linear projections for route and speed,
    plus a shared learned positional embedding, uniform-initialized to match
    TFv6's token init convention.

    When route correction is active (eff_rank > 0): produces 2 tokens —
    route_tok [B, 1, D] and speed_tok [B, 1, D].
    When route correction is disabled (eff_rank == 0): produces 1 token —
    speed_tok [B, 1, D].
    """

    def __init__(self, eff_rank: int, d_model: int):
        super().__init__()
        self.eff_rank = eff_rank
        if eff_rank > 0:
            self.route_encoder = nn.Linear(eff_rank, d_model)
        self.speed_encoder = nn.Linear(1, d_model)
        n_tokens = 2 if eff_rank > 0 else 1
        self.pos_embedding = nn.Parameter(torch.zeros(1, n_tokens, d_model))
        nn.init.uniform_(self.pos_embedding)

    def forward(self, action_coeffs: torch.Tensor) -> torch.Tensor:
        # action_coeffs: [B, eff_rank + 1]
        speed_tok = self.speed_encoder(
            action_coeffs[:, self.eff_rank :].float()
        ).unsqueeze(1)  # [B, 1, D]
        if self.eff_rank > 0:
            route_tok = self.route_encoder(
                action_coeffs[:, : self.eff_rank].float()
            ).unsqueeze(1)  # [B, 1, D]
            tokens = torch.cat([route_tok, speed_tok], dim=1)  # [B, 2, D]
        else:
            tokens = speed_tok  # [B, 1, D]
        return tokens + self.pos_embedding


# ── Critic privileged token encoder ──────────────────────────────────────────


class CriticPrivilegedTokenEncoder(nn.Module):
    """Encode privileged measurements as a single KV token (critic-only).

    Mirrors CriticActionTokenEncoder: Linear projection + learned positional
    embedding, uniform-initialized. Privileged info (stop sign distance,
    traffic light state, etc.) is critic-only — not available at inference.
    """

    def __init__(self, priv_dim: int, d_model: int):
        super().__init__()
        self.encoder = nn.Linear(priv_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.uniform_(self.pos_embedding)

    def forward(self, privileged: torch.Tensor) -> torch.Tensor:
        return self.encoder(privileged.float()).unsqueeze(1) + self.pos_embedding


# ── Critic decoder ────────────────────────────────────────────────────────────


class CriticDecoder(_ResidualTransformerDecoder):
    """Critic head: action (+ optional privileged) tokens appended to KV → scalar.

    action_coeffs are encoded via CriticActionTokenEncoder (1 or 2 tokens).
    privileged measurements are encoded via CriticPrivilegedTokenEncoder (1 token).
    Both are appended to the KV sequence before decoding so the decoder can
    attend to them in context. The decoded primary query goes directly to the
    Q head — no concatenation after decoding.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        n_queries: int,
        ffn_dim: int,
        eff_rank: int,
        priv_dim: int = 0,
        concat_action_after_decoder: bool = False,
    ):
        super().__init__(d_model, n_heads, n_layers, n_queries, ffn_dim)
        self.action_encoder = CriticActionTokenEncoder(eff_rank, d_model)
        self.priv_encoder = (
            CriticPrivilegedTokenEncoder(priv_dim, d_model) if priv_dim > 0 else None
        )
        self.kv_norm = nn.LayerNorm(d_model)
        self.concat_action_after_decoder = concat_action_after_decoder
        q_head_in = d_model + (eff_rank + 1) if concat_action_after_decoder else d_model
        self.q_head = nn.Sequential(
            nn.LayerNorm(q_head_in),
            nn.Linear(q_head_in, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        kv_tokens: torch.Tensor,
        action_coeffs: torch.Tensor,
        privileged: torch.Tensor | None = None,
    ) -> torch.Tensor:
        action_tok = self.action_encoder(action_coeffs.float())  # [B, 1 or 2, D]
        kv = torch.cat([kv_tokens, action_tok], dim=1)
        if self.priv_encoder is not None and privileged is not None:
            priv_tok = self.priv_encoder(privileged)  # [B, 1, D]
            kv = torch.cat([kv, priv_tok], dim=1)
        decoded = self._decode(self.kv_norm(kv))  # [B, N_queries, D]
        q_input = (
            torch.cat([decoded[:, 0], action_coeffs.float()], dim=-1)
            if self.concat_action_after_decoder
            else decoded[:, 0]
        )
        return self.q_head(q_input)  # [B, 1]


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
        kv_tokens = self.backbone.get_residual_features(
            base_action_mean, obs_dict.get("speed_history")
        )
        if detach_features:
            kv_tokens = kv_tokens.detach()
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
            self.backbone._last_base_action_mean,
            feature_obs.get("speed_history"),
        )
        if detach_features:
            kv_tokens = kv_tokens.detach()
        return self.residual_out(kv_tokens)

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

    self.backbone = TFv6TransformerBackbone (plain reference, NOT a child module).
    self.q_head   = CriticDecoder (used by critic_optimizer).

    The trainer owns the shared backbone and passes backbone.encoder_parameters()
    to the critic optimizer. This keeps qf1/qf2 state_dicts and parameter lists
    limited to their independent Q heads.

    Interface is identical to TFv6ResidualQNetworkTD3.
    """

    def __init__(
        self,
        backbone: TFv6TransformerBackbone,
        rl_config=None,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "_backbone_ref", backbone)

        eff_rank = (
            0 if backbone.disable_residual_route else backbone.residual_route_rank
        )

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
        self.q_head = CriticDecoder(
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            n_queries=N_QUERIES,
            ffn_dim=FFN_DIM,
            eff_rank=eff_rank,
            priv_dim=priv_dim,
            concat_action_after_decoder=bool(
                getattr(
                    rl_config,
                    "transformer_concat_coef_action_to_head_after_decoder",
                    False,
                )
            ),
        )

    @property
    def backbone(self) -> TFv6TransformerBackbone:
        return object.__getattribute__(self, "_backbone_ref")

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
        kv_tokens = self.backbone.get_residual_features(base_action, speed_history)
        return self.q_head(kv_tokens, action_coeffs, privileged)

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
        kv_tokens = self.backbone.get_residual_features(base_action, speed_history)
        if detach_features:
            kv_tokens = kv_tokens.detach()
        return self.q_head(kv_tokens, action_coeffs, privileged)
