from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Beta, Categorical, MultivariateNormal, Normal

from rl_finetuning.tfv6_rl.action_codec import ActionCodec


def _sum_independent_dims(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim > 1:
        return tensor.sum(dim=1)
    return tensor.sum()


class DiagGaussianDistribution(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.distribution: Normal | None = None

    def proba_distribution(
        self, mean_actions: torch.Tensor, log_std: torch.Tensor
    ) -> DiagGaussianDistribution:
        self.distribution = Normal(mean_actions, torch.exp(log_std))
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        assert self.distribution is not None
        return _sum_independent_dims(self.distribution.log_prob(actions))

    def entropy(self) -> torch.Tensor:
        assert self.distribution is not None
        return self.distribution.entropy()

    def sample(self) -> torch.Tensor:
        assert self.distribution is not None
        return self.distribution.rsample()

    def mode(self) -> torch.Tensor:
        assert self.distribution is not None
        return self.distribution.mean

    def get_actions(self, sample_type: str = "sample") -> torch.Tensor:
        if sample_type in ("mean", "mode", "deterministic"):
            return self.mode()
        return self.sample()

    def exploration_loss(self, *_args, **_kwargs) -> torch.Tensor:
        assert self.distribution is not None
        return torch.zeros((), device=self.distribution.mean.device)


class CorrelatedGaussianDistribution(nn.Module):
    """Correlated Gaussian with block-diagonal covariance along route/waypoints."""

    def __init__(
        self, action_codec: ActionCodec, rho: float = 0.8, jitter: float = 1e-6
    ) -> None:
        super().__init__()
        self.action_codec = action_codec
        self.rho = float(rho)
        self.jitter = float(jitter)
        self.distribution: MultivariateNormal | None = None

    def _corr_matrix(
        self, n: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        if n <= 1:
            return torch.ones((n, n), device=device, dtype=dtype)
        idx = torch.arange(n, device=device)
        dist = (idx[None, :] - idx[:, None]).abs()
        return (self.rho**dist).to(dtype=dtype)

    def _build_cov_block(
        self, std_slice: torch.Tensor, n_points: int, device, dtype
    ) -> torch.Tensor:
        if n_points == 0:
            return torch.zeros((0, 0), device=device, dtype=dtype)

        std_x = std_slice[0::2]
        std_y = std_slice[1::2]
        corr = self._corr_matrix(n_points, device=device, dtype=dtype)

        cov_x = torch.outer(std_x, std_x) * corr
        cov_y = torch.outer(std_y, std_y) * corr

        cov = torch.zeros((2 * n_points, 2 * n_points), device=device, dtype=dtype)
        cov[0::2, 0::2] = cov_x
        cov[1::2, 1::2] = cov_y
        cov = cov + torch.eye(2 * n_points, device=device, dtype=dtype) * self.jitter
        return cov

    def proba_distribution(
        self, mean_actions: torch.Tensor, log_std: torch.Tensor
    ) -> CorrelatedGaussianDistribution:
        std = torch.exp(log_std)
        device = mean_actions.device
        dtype = mean_actions.dtype
        batch_size = mean_actions.shape[0]

        scale_trils = []
        for i in range(batch_size):
            std_row = std[i]
            blocks = []
            if self.action_codec.predict_route:
                route_std = std_row[self.action_codec.slices.route]
                blocks.append(
                    self._build_cov_block(
                        route_std, self.action_codec.num_route_points, device, dtype
                    )
                )
            if self.action_codec.predict_waypoints:
                wp_std = std_row[self.action_codec.slices.waypoints]
                blocks.append(
                    self._build_cov_block(
                        wp_std, self.action_codec.num_waypoints, device, dtype
                    )
                )
            if self.action_codec.predict_target_speed:
                speed_std = std_row[self.action_codec.slices.target_speed]
                blocks.append(
                    speed_std.pow(2).reshape(1, 1)
                    + torch.eye(1, device=device, dtype=dtype) * self.jitter
                )
            cov = blocks[0] if len(blocks) == 1 else torch.block_diag(*blocks)
            scale_trils.append(torch.linalg.cholesky(cov))
        self.distribution = MultivariateNormal(
            loc=mean_actions, scale_tril=torch.stack(scale_trils, dim=0)
        )
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        assert self.distribution is not None
        return self.distribution.log_prob(actions)

    def entropy(self) -> torch.Tensor:
        assert self.distribution is not None
        return self.distribution.entropy()

    def sample(self) -> torch.Tensor:
        assert self.distribution is not None
        return self.distribution.rsample()

    def mode(self) -> torch.Tensor:
        assert self.distribution is not None
        return self.distribution.mean

    def get_actions(self, sample_type: str = "sample") -> torch.Tensor:
        if sample_type in ("mean", "mode", "deterministic"):
            return self.mode()
        return self.sample()

    def exploration_loss(self, *_args, **_kwargs) -> torch.Tensor:
        assert self.distribution is not None
        return torch.zeros((), device=self.distribution.mean.device)


class LowRankDiagRouteGaussianDistribution(nn.Module):
    """Gaussian with route block covariance = low-rank smooth modes + diagonal."""

    def __init__(
        self,
        action_codec: ActionCodec,
        route_lowrank_rank: int = 6,
        route_lowrank_std: float = 0.015,
        jitter: float = 1e-6,
    ) -> None:
        super().__init__()
        self.action_codec = action_codec
        self.route_lowrank_rank = max(1, int(route_lowrank_rank))
        self.route_lowrank_std = float(route_lowrank_std)
        self.jitter = float(jitter)
        self.distribution: MultivariateNormal | None = None
        self._wp_std: torch.Tensor | None = (
            None  # stored in proba_distribution for log_prob
        )

        if (
            not self.action_codec.predict_route
            or self.action_codec.slices.route is None
        ):
            raise ValueError(
                "lowrank_diag_route_sampling requires route predictions to be enabled."
            )
        route_dim = int(self.action_codec.route_dim)
        if route_dim <= 0:
            raise ValueError("Route action dimension must be positive.")
        self.route_lowrank_rank = min(self.route_lowrank_rank, route_dim)
        basis = self._build_route_basis(
            num_route_points=self.action_codec.num_route_points,
            route_dim=route_dim,
            rank=self.route_lowrank_rank,
        )
        self.register_buffer("route_basis", basis, persistent=False)

    def _build_route_basis(
        self, *, num_route_points: int, route_dim: int, rank: int
    ) -> torch.Tensor:
        # Smooth low-rank route modes over normalized path coordinate u in [0, 1].
        # We prioritize two interpretable lateral modes:
        # 1) u*sin(pi*u/2): broad turn-like bending with movable tail
        # 2) u*(1-u): lane-change-like mid-route bulge.
        u = torch.linspace(0.0, 1.0, steps=max(num_route_points, 2))[:num_route_points]
        cols: list[torch.Tensor] = []

        def _add_axis_mode(mode_values: torch.Tensor, axis: int) -> None:
            if len(cols) >= rank:
                return
            col = torch.zeros(route_dim, dtype=torch.float32)
            col[axis::2] = mode_values
            col = col / (torch.linalg.norm(col) + 1e-8)
            cols.append(col)

        turn_mode = u * torch.sin(0.5 * math.pi * u)
        lane_change_mode = u * (1.0 - u)

        # Keep first two modes as requested and lateral first for controller relevance.
        _add_axis_mode(turn_mode, axis=1)  # y
        _add_axis_mode(lane_change_mode, axis=1)  # y

        # If more rank is requested, add longitudinal counterparts.
        _add_axis_mode(turn_mode, axis=0)  # x
        _add_axis_mode(lane_change_mode, axis=0)  # x

        # Extra fallback modes if rank > 4.
        mode = 1
        while len(cols) < rank:
            extra = u * torch.sin((mode + 0.5) * math.pi * u)
            _add_axis_mode(extra, axis=1)  # y
            _add_axis_mode(extra, axis=0)  # x
            mode += 1

        return torch.stack(cols, dim=1)  # [route_dim, rank]

    def _route_covariance(
        self, route_std: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        cov = torch.diag(route_std.pow(2))
        if self.route_lowrank_std > 0.0:
            basis = self.route_basis.to(device=device, dtype=dtype)
            lowrank = basis * self.route_lowrank_std
            cov = cov + lowrank @ lowrank.transpose(0, 1)
        cov = cov + torch.eye(cov.shape[0], device=device, dtype=dtype) * self.jitter
        return cov

    def proba_distribution(
        self, mean_actions: torch.Tensor, log_std: torch.Tensor
    ) -> LowRankDiagRouteGaussianDistribution:
        std = torch.exp(log_std)
        device = mean_actions.device
        dtype = mean_actions.dtype
        batch_size = mean_actions.shape[0]
        route_slice = self.action_codec.slices.route

        # Store waypoint stds for the projected log_prob computation.
        self._wp_std = (
            std[:, self.action_codec.slices.waypoints].clone()
            if self.action_codec.predict_waypoints
            else None
        )

        scale_trils = []
        for i in range(batch_size):
            std_row = std[i]
            blocks = []
            if route_slice is not None:
                route_std = std_row[route_slice]
                blocks.append(self._route_covariance(route_std, device, dtype))
            if self.action_codec.predict_waypoints:
                wp_std = std_row[self.action_codec.slices.waypoints]
                wp_dim = wp_std.shape[0]
                blocks.append(
                    torch.diag(wp_std.pow(2))
                    + torch.eye(wp_dim, device=device, dtype=dtype) * self.jitter
                )
            if self.action_codec.predict_target_speed:
                speed_std = std_row[self.action_codec.slices.target_speed]
                blocks.append(
                    speed_std.pow(2).reshape(1, 1)
                    + torch.eye(1, device=device, dtype=dtype) * self.jitter
                )
            cov = blocks[0] if len(blocks) == 1 else torch.block_diag(*blocks)
            scale_trils.append(torch.linalg.cholesky(cov))
        self.distribution = MultivariateNormal(
            loc=mean_actions, scale_tril=torch.stack(scale_trils, dim=0)
        )
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        assert self.distribution is not None
        # Route: evaluate log_prob only in the low-rank basis subspace.
        # This avoids the precision explosion in the (route_dim - rank) orthogonal
        # directions, which have variance ~jitter=1e-6 and would cause PPO ratios
        # to explode for any small mean shift in those directions.
        route_slice = self.action_codec.slices.route
        residual_route = actions[:, route_slice] - self.distribution.loc[:, route_slice]
        basis = self.route_basis.to(dtype=actions.dtype, device=actions.device)
        proj = residual_route @ basis  # [B, rank] — amplitudes along smooth modes
        log_prob = (
            Normal(
                torch.zeros_like(proj),
                proj.new_full(proj.shape, self.route_lowrank_std),
            )
            .log_prob(proj)
            .sum(dim=1)
        )
        # Waypoints: standard diagonal Gaussian (no precision explosion there).
        if self._wp_std is not None:
            wp_slice = self.action_codec.slices.waypoints
            residual_wp = actions[:, wp_slice] - self.distribution.loc[:, wp_slice]
            log_prob = log_prob + Normal(
                torch.zeros_like(residual_wp),
                self._wp_std.to(dtype=residual_wp.dtype),
            ).log_prob(residual_wp).sum(dim=1)
        return log_prob

    def entropy(self) -> torch.Tensor:
        assert self.distribution is not None
        # Match the projected log_prob: entropy of the rank-dim route Gaussian
        # plus diagonal waypoint entropy.
        device = self.distribution.loc.device
        dtype = self.distribution.loc.dtype
        batch_size = self.distribution.loc.shape[0]
        route_ent = (
            0.5
            * math.log(2.0 * math.pi * math.e * self.route_lowrank_std**2)
            * self.route_lowrank_rank
        )
        result = torch.full((batch_size,), route_ent, device=device, dtype=dtype)
        if self._wp_std is not None:
            wp_entropy = (
                Normal(torch.zeros_like(self._wp_std), self._wp_std)
                .entropy()
                .sum(dim=1)
            )
            result = result + wp_entropy
        return result

    def sample(self) -> torch.Tensor:
        assert self.distribution is not None
        return self.distribution.rsample()

    def mode(self) -> torch.Tensor:
        assert self.distribution is not None
        return self.distribution.mean

    def get_actions(self, sample_type: str = "sample") -> torch.Tensor:
        if sample_type in ("mean", "mode", "deterministic"):
            return self.mode()
        return self.sample()

    def exploration_loss(self, *_args, **_kwargs) -> torch.Tensor:
        assert self.distribution is not None
        return torch.zeros((), device=self.distribution.mean.device)


class DiagBetaDistribution(nn.Module):
    """Independent Beta per action dimension on [-1, 1], parameterized via mean + concentration."""

    def __init__(
        self,
        eps: float = 1e-4,
        concentration_min: float = 2.0,
        concentration_max: float = 200.0,
    ) -> None:
        super().__init__()
        self.eps = float(eps)
        self.concentration_min = float(concentration_min)
        self.concentration_max = float(concentration_max)
        self.distribution: Beta | None = None
        self._mean_actions: torch.Tensor | None = None

    def proba_distribution(
        self, mean_actions: torch.Tensor, concentration: torch.Tensor
    ) -> DiagBetaDistribution:
        mean_actions = torch.clamp(mean_actions, -1.0 + self.eps, 1.0 - self.eps)
        mean01 = torch.clamp((mean_actions + 1.0) * 0.5, self.eps, 1.0 - self.eps)
        concentration = torch.clamp(
            concentration,
            min=self.concentration_min + self.eps,
            max=self.concentration_max,
        )
        alpha = torch.clamp(mean01 * concentration, min=self.eps)
        beta = torch.clamp((1.0 - mean01) * concentration, min=self.eps)
        self.distribution = Beta(alpha, beta)
        self._mean_actions = mean_actions
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        assert self.distribution is not None
        x = torch.clamp((actions + 1.0) * 0.5, self.eps, 1.0 - self.eps)
        # dy/dx = 2 -> p(y)=p(x)/2, so log p(y)=log p(x)-log 2
        log_prob = self.distribution.log_prob(x) - torch.log(
            torch.tensor(2.0, device=x.device, dtype=x.dtype)
        )
        return _sum_independent_dims(log_prob)

    def entropy(self) -> torch.Tensor:
        assert self.distribution is not None
        ent = self.distribution.entropy() + torch.log(
            torch.tensor(
                2.0,
                device=self.distribution.concentration0.device,
                dtype=self.distribution.concentration0.dtype,
            )
        )
        return ent

    def sample(self) -> torch.Tensor:
        assert self.distribution is not None
        x = self.distribution.rsample()
        return x * 2.0 - 1.0

    def mode(self) -> torch.Tensor:
        # Use mean for deterministic mode; robust even when alpha/beta <= 1.
        assert self.distribution is not None
        mean01 = self.distribution.mean
        return mean01 * 2.0 - 1.0

    def get_actions(self, sample_type: str = "sample") -> torch.Tensor:
        if sample_type in ("mean", "mode", "deterministic"):
            return self.mode()
        return self.sample()

    def exploration_loss(self, *_args, **_kwargs) -> torch.Tensor:
        assert self.distribution is not None
        return torch.zeros((), device=self.distribution.mean.device)


@dataclass
class NoiseDiagnostics:
    noise_pred: torch.Tensor
    diag_std: torch.Tensor
    diag_log_std: torch.Tensor
    path_amp1_std: torch.Tensor | None = None
    path_amp2_std: torch.Tensor | None = None


class HybridActionDistribution:
    """Continuous route/waypoint distribution plus native categorical speed bins."""

    def __init__(
        self,
        continuous_dist: nn.Module | None,
        speed_logits: torch.Tensor,
        speed_values: torch.Tensor,
        continuous_action_dim: int,
    ) -> None:
        self.continuous_dist = continuous_dist
        self.speed_categorical = Categorical(logits=speed_logits.float())
        self.speed_values = speed_values.to(
            device=speed_logits.device, dtype=speed_logits.dtype
        )
        self.continuous_action_dim = int(continuous_action_dim)
        self.distribution = self

    def _split_actions(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        if actions.ndim == 1:
            actions = actions.unsqueeze(0)
        continuous_actions = None
        if self.continuous_action_dim > 0:
            continuous_actions = actions[:, : self.continuous_action_dim]
        speed_actions = actions[:, self.continuous_action_dim]
        return continuous_actions, speed_actions

    def _compose_actions(
        self, continuous_actions: torch.Tensor | None, speed_actions: torch.Tensor
    ) -> torch.Tensor:
        if continuous_actions is None:
            return speed_actions.unsqueeze(1)
        return torch.cat((continuous_actions, speed_actions.unsqueeze(1)), dim=1)

    def _speed_indices(self, speed_actions: torch.Tensor) -> torch.Tensor:
        if speed_actions.ndim == 2:
            speed_actions = speed_actions.squeeze(1)
        diffs = (speed_actions.unsqueeze(1) - self.speed_values.unsqueeze(0)).abs()
        return diffs.argmin(dim=1)

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        continuous_actions, speed_actions = self._split_actions(actions)
        log_prob = torch.zeros(
            speed_actions.shape[0],
            device=speed_actions.device,
            dtype=speed_actions.dtype,
        )
        if continuous_actions is not None and self.continuous_dist is not None:
            log_prob = log_prob + self.continuous_dist.log_prob(continuous_actions).to(
                log_prob.dtype
            )
        speed_indices = self._speed_indices(speed_actions)
        log_prob = log_prob + self.speed_categorical.log_prob(speed_indices).to(
            log_prob.dtype
        )
        return log_prob

    def entropy(self) -> torch.Tensor:
        entropy = self.speed_categorical.entropy()
        if self.continuous_dist is not None:
            continuous_entropy = self.continuous_dist.entropy()
            if continuous_entropy.ndim > 1:
                continuous_entropy = continuous_entropy.sum(dim=1)
            entropy = entropy.to(continuous_entropy.dtype) + continuous_entropy
        return entropy

    def sample(self) -> torch.Tensor:
        continuous_actions = None
        if self.continuous_dist is not None:
            continuous_actions = self.continuous_dist.sample()
        speed_indices = self.speed_categorical.sample()
        speed_actions = self.speed_values[speed_indices]
        return self._compose_actions(continuous_actions, speed_actions)

    def mode(self) -> torch.Tensor:
        continuous_actions = None
        if self.continuous_dist is not None:
            continuous_actions = self.continuous_dist.mode()
        speed_probs = self.speed_categorical.probs.to(self.speed_values.dtype)
        speed_actions = (speed_probs * self.speed_values.unsqueeze(0)).sum(dim=1)
        return self._compose_actions(continuous_actions, speed_actions)

    def get_actions(self, sample_type: str = "sample") -> torch.Tensor:
        if sample_type in ("mean", "mode", "deterministic"):
            return self.mode()
        return self.sample()

    def exploration_loss(self, *_args, **_kwargs) -> torch.Tensor:
        return torch.zeros(
            (),
            device=self.speed_categorical.logits.device,
            dtype=self.speed_values.dtype,
        )

    def kl_divergence(self, other: HybridActionDistribution) -> torch.Tensor:
        kl_terms = []
        if self.continuous_dist is not None and other.continuous_dist is not None:
            continuous_kl = torch.distributions.kl_divergence(
                self.continuous_dist.distribution, other.continuous_dist.distribution
            )
            if continuous_kl.ndim > 1:
                continuous_kl = continuous_kl.sum(dim=1)
            kl_terms.append(continuous_kl)
        categorical_kl = torch.distributions.kl_divergence(
            self.speed_categorical, other.speed_categorical
        )
        kl_terms.append(categorical_kl)
        return torch.stack(kl_terms, dim=0).sum(dim=0)


class ActionNoiseCodec:
    """Owns noise-head parameterization (grouped) and distribution construction."""

    SAMPLER_SPLINE_CURVATURE = "spline_curvature_perturbation"
    SAMPLER_LEGACY = "legacy_noise_sampling"
    SAMPLER_LOWRANK_DIAG_ROUTE = "lowrank_diag_route_sampling"

    def __init__(
        self,
        action_codec: ActionCodec,
        *,
        distribution_type: str = "gaussian",
        sampling_technique: str = SAMPLER_SPLINE_CURVATURE,
        use_correlated_gaussian: bool = True,
        correlated_noise_rho: float = 0.8,
        noise_ramp: bool = True,
        log_std_min: float = -5.0,
        log_std_max: float = 1.0,
        heading_amplitude1_std_init: float = 0.07,
        heading_amplitude2_std_init: float = 0.03,
        path_std_base_frac: float = 0.15,
        lowrank_route_rank: int = 6,
        lowrank_route_std_init: float = 0.015,
        beta_eps: float = 1e-4,
        beta_concentration_min: float = 2.0,
        beta_concentration_max: float = 200.0,
        speed_sampling_style: str = "native_categorical",
    ) -> None:
        self.full_action_codec = action_codec
        self.distribution_type = str(distribution_type)
        self.sampling_technique = str(sampling_technique)
        self.use_correlated_gaussian = bool(use_correlated_gaussian)
        self.correlated_noise_rho = float(correlated_noise_rho)
        self.noise_ramp = bool(noise_ramp)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.heading_amplitude1_std_init = float(heading_amplitude1_std_init)
        self.heading_amplitude2_std_init = float(heading_amplitude2_std_init)
        self.path_std_base_frac = float(path_std_base_frac)
        self.lowrank_route_rank = int(lowrank_route_rank)
        self.lowrank_route_std_init = float(lowrank_route_std_init)
        self.beta_eps = float(beta_eps)
        self.beta_concentration_min = float(beta_concentration_min)
        self.beta_concentration_max = float(beta_concentration_max)
        self.speed_sampling_style = str(speed_sampling_style)
        if self.speed_sampling_style == "mean_gassian":
            self.speed_sampling_style = "mean_gaussian"
        if self.speed_sampling_style not in ("mean_gaussian", "native_categorical"):
            raise ValueError(
                "speed_sampling_style must be one of "
                "{'mean_gaussian', 'native_categorical'}"
            )
        self.use_native_categorical_speed = bool(
            self.speed_sampling_style == "native_categorical"
            and self.full_action_codec.predict_target_speed
        )
        self.action_codec = self.full_action_codec
        if self.use_native_categorical_speed:
            self.action_codec = ActionCodec(
                self.full_action_codec.config,
                use_route=self.full_action_codec.predict_route,
                use_waypoints=self.full_action_codec.predict_waypoints,
                use_target_speed=False,
            )
            self.speed_values = torch.tensor(
                self.full_action_codec.config.target_speed_classes,
                dtype=torch.float32,
            ) / float(self.full_action_codec.config.max_speed)
        else:
            self.speed_values = None
        self.continuous_action_dim = int(self.action_codec.action_dim)
        self.full_action_dim = int(self.full_action_codec.action_dim)

        self.group_names: list[str] = []
        self.group_slices: list[slice] = []
        if self.action_codec.predict_route:
            self.group_names.append("route")
            self.group_slices.append(self.action_codec.slices.route)
        if self.action_codec.predict_waypoints:
            self.group_names.append("waypoints")
            self.group_slices.append(self.action_codec.slices.waypoints)
        if self.action_codec.predict_target_speed:
            self.group_names.append("target_speed")
            self.group_slices.append(self.action_codec.slices.target_speed)
        if not self.group_names and not self.use_native_categorical_speed:
            raise RuntimeError("No active action heads found for ActionNoiseCodec.")

        if self.sampling_technique not in (
            self.SAMPLER_SPLINE_CURVATURE,
            self.SAMPLER_LEGACY,
            self.SAMPLER_LOWRANK_DIAG_ROUTE,
        ):
            raise ValueError(
                f"Unsupported sampling technique: {self.sampling_technique}"
            )
        if not (0.0 <= self.path_std_base_frac <= 1.0):
            raise ValueError(
                f"path_std_base_frac must be in [0,1], got {self.path_std_base_frac}"
            )
        if self.heading_amplitude1_std_init < 0.0:
            raise ValueError(
                f"heading_amplitude1_std_init must be >= 0, got {self.heading_amplitude1_std_init}"
            )
        if self.heading_amplitude2_std_init < 0.0:
            raise ValueError(
                f"heading_amplitude2_std_init must be >= 0, got {self.heading_amplitude2_std_init}"
            )
        if self.lowrank_route_rank < 1:
            raise ValueError(
                f"lowrank_route_rank must be >= 1, got {self.lowrank_route_rank}"
            )
        if self.lowrank_route_std_init < 0.0:
            raise ValueError(
                f"lowrank_route_std_init must be >= 0, got {self.lowrank_route_std_init}"
            )

        self.base_noise_pred_dim = len(self.group_names)
        self.route_group_index = (
            self.group_names.index("route") if "route" in self.group_names else None
        )
        self.use_spline_curvature_sampler = (
            self.distribution_type == "gaussian"
            and self.sampling_technique == self.SAMPLER_SPLINE_CURVATURE
            and self.route_group_index is not None
        )
        self.use_lowrank_diag_route_sampler = (
            self.distribution_type == "gaussian"
            and self.sampling_technique == self.SAMPLER_LOWRANK_DIAG_ROUTE
            and self.route_group_index is not None
        )

        if self.distribution_type == "gaussian":
            self.noise_pred_dim = self.base_noise_pred_dim + (
                2 if self.use_spline_curvature_sampler else 0
            )
            if self.base_noise_pred_dim == 0:
                self.action_dist = None
            elif self.use_lowrank_diag_route_sampler:
                self.action_dist = LowRankDiagRouteGaussianDistribution(
                    self.action_codec,
                    route_lowrank_rank=self.lowrank_route_rank,
                    route_lowrank_std=self.lowrank_route_std_init,
                )
            elif self.use_correlated_gaussian:
                self.action_dist = CorrelatedGaussianDistribution(
                    self.action_codec, rho=self.correlated_noise_rho
                )
            else:
                self.action_dist = DiagGaussianDistribution()
        elif self.distribution_type == "beta":
            if self.sampling_technique != self.SAMPLER_LEGACY:
                raise ValueError(
                    "Beta action noise currently supports only legacy_noise_sampling."
                )
            self.noise_pred_dim = len(
                self.group_names
            )  # one concentration per active head type
            if self.base_noise_pred_dim == 0:
                self.action_dist = None
            else:
                self.action_dist = DiagBetaDistribution(
                    eps=self.beta_eps,
                    concentration_min=self.beta_concentration_min,
                    concentration_max=self.beta_concentration_max,
                )
        else:
            raise ValueError(
                f"Unsupported action noise distribution: {self.distribution_type}"
            )
        self.noise_pred_names = list(self.group_names)
        if self.use_spline_curvature_sampler:
            self.noise_pred_names.extend(
                ("heading_amplitude1_log_std", "heading_amplitude2_log_std")
            )

    def _pad_speed_diagnostics(self, diag: NoiseDiagnostics) -> NoiseDiagnostics:
        if not self.use_native_categorical_speed:
            return diag
        batch_size = diag.diag_log_std.shape[0]
        speed_log_std = torch.zeros(
            (batch_size, 1),
            device=diag.diag_log_std.device,
            dtype=diag.diag_log_std.dtype,
        )
        speed_std = torch.ones_like(speed_log_std)
        return NoiseDiagnostics(
            noise_pred=diag.noise_pred,
            diag_std=torch.cat((diag.diag_std, speed_std), dim=1),
            diag_log_std=torch.cat((diag.diag_log_std, speed_log_std), dim=1),
            path_amp1_std=diag.path_amp1_std,
            path_amp2_std=diag.path_amp2_std,
        )

    def expand_group_values(self, grouped_values: torch.Tensor) -> torch.Tensor:
        if grouped_values.ndim != 2 or grouped_values.shape[1] != len(self.group_names):
            raise ValueError(
                f"Expected grouped values [B,{len(self.group_names)}], got {tuple(grouped_values.shape)}"
            )
        expanded = torch.empty(
            grouped_values.shape[0],
            self.action_codec.action_dim,
            device=grouped_values.device,
            dtype=grouped_values.dtype,
        )
        for idx, sl in enumerate(self.group_slices):
            expanded[:, sl] = grouped_values[:, idx].unsqueeze(1)
        return expanded

    def _apply_group_noise_ramp(self, grouped_log_std: torch.Tensor) -> torch.Tensor:
        grouped_log_std = torch.clamp(
            grouped_log_std, self.log_std_min, self.log_std_max
        )
        if not self.noise_ramp:
            return grouped_log_std
        std = torch.exp(grouped_log_std)
        scaled = std.clone()
        for idx, name in enumerate(self.group_names):
            if name in ("route", "waypoints"):
                scaled[:, idx] = scaled[:, idx] * 0.5
        return torch.log(scaled + 1e-6)

    def _split_noise_pred(
        self, noise_pred: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if noise_pred.ndim != 2:
            raise ValueError(
                f"Expected noise_pred [B,D], got {tuple(noise_pred.shape)}"
            )
        if noise_pred.shape[1] != self.noise_pred_dim:
            raise ValueError(
                f"Expected noise_pred dim {self.noise_pred_dim}, got {noise_pred.shape[1]}"
            )
        base_pred = noise_pred[:, : self.base_noise_pred_dim]
        if not self.use_spline_curvature_sampler:
            return base_pred, None, None
        amp1 = torch.clamp(
            noise_pred[:, self.base_noise_pred_dim], self.log_std_min, self.log_std_max
        )
        amp2 = torch.clamp(
            noise_pred[:, self.base_noise_pred_dim + 1],
            self.log_std_min,
            self.log_std_max,
        )
        return base_pred, amp1, amp2

    def _join_noise_pred(
        self,
        base_pred: torch.Tensor,
        amp1_log_std: torch.Tensor | None,
        amp2_log_std: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.use_spline_curvature_sampler:
            return base_pred
        if amp1_log_std is None or amp2_log_std is None:
            raise ValueError(
                "Spline curvature sampling expects heading amplitude log-std values."
            )
        return torch.cat(
            (base_pred, amp1_log_std.unsqueeze(1), amp2_log_std.unsqueeze(1)), dim=1
        )

    def gaussian_diagnostics_from_head(
        self, head_pred: torch.Tensor
    ) -> NoiseDiagnostics:
        base_pred, amp1_log_std, amp2_log_std = self._split_noise_pred(head_pred)
        grouped_log_std = self._apply_group_noise_ramp(base_pred)
        diag_log_std = self.expand_group_values(grouped_log_std)
        diag_std = torch.exp(diag_log_std)
        combined_noise_pred = self._join_noise_pred(
            grouped_log_std, amp1_log_std, amp2_log_std
        )
        return NoiseDiagnostics(
            noise_pred=combined_noise_pred,
            diag_std=diag_std,
            diag_log_std=diag_log_std,
            path_amp1_std=None
            if amp1_log_std is None
            else torch.exp(amp1_log_std).to(diag_std.dtype),
            path_amp2_std=None
            if amp2_log_std is None
            else torch.exp(amp2_log_std).to(diag_std.dtype),
        )

    def gaussian_diagnostics_from_noise_pred(
        self, noise_pred: torch.Tensor
    ) -> NoiseDiagnostics:
        base_pred, amp1_log_std, amp2_log_std = self._split_noise_pred(noise_pred)
        diag_log_std = self.expand_group_values(base_pred)
        diag_std = torch.exp(diag_log_std)
        combined_noise_pred = self._join_noise_pred(
            base_pred, amp1_log_std, amp2_log_std
        )
        return NoiseDiagnostics(
            noise_pred=combined_noise_pred,
            diag_std=diag_std,
            diag_log_std=diag_log_std,
            path_amp1_std=None
            if amp1_log_std is None
            else torch.exp(amp1_log_std).to(diag_std.dtype),
            path_amp2_std=None
            if amp2_log_std is None
            else torch.exp(amp2_log_std).to(diag_std.dtype),
        )

    def beta_diagnostics_from_head(
        self, mean_actions: torch.Tensor, head_pred: torch.Tensor
    ) -> NoiseDiagnostics:
        grouped_concentration = (
            torch.nn.functional.softplus(head_pred) + self.beta_concentration_min
        )
        return self.beta_diagnostics_from_noise_pred(
            mean_actions, grouped_concentration
        )

    def beta_diagnostics_from_noise_pred(
        self, mean_actions: torch.Tensor, noise_pred: torch.Tensor
    ) -> NoiseDiagnostics:
        grouped_concentration = noise_pred
        diag_concentration = self.expand_group_values(grouped_concentration)
        mean_actions = torch.clamp(mean_actions, -1.0 + 1e-4, 1.0 - 1e-4)
        mean01 = torch.clamp((mean_actions + 1.0) * 0.5, 1e-4, 1.0 - 1e-4)
        alpha = mean01 * diag_concentration
        beta = (1.0 - mean01) * diag_concentration
        var01 = (alpha * beta) / ((alpha + beta).pow(2) * (alpha + beta + 1.0))
        diag_std = torch.sqrt(torch.clamp(4.0 * var01, min=1e-12))
        diag_log_std = torch.log(diag_std + 1e-12)
        return NoiseDiagnostics(
            noise_pred=grouped_concentration,
            diag_std=diag_std,
            diag_log_std=diag_log_std,
        )

    def _softplus_inverse(self, x: float) -> float:
        if x <= 0.0:
            return -20.0
        if x > 20.0:
            return x
        return math.log(math.expm1(x))

    def default_head_bias_from_log_std_init(self, log_std_init: float) -> float:
        """Map a desired Gaussian-style log-std target to this codec's head preactivation."""
        if self.distribution_type == "gaussian":
            return float(log_std_init)

        # For Beta on [-1, 1], match the center-point symmetric Beta std:
        # std_center = 1 / sqrt(concentration + 1), then clip to feasible concentration.
        target_std = max(math.exp(float(log_std_init)), 1e-8)
        target_concentration = (1.0 / (target_std * target_std)) - 1.0
        target_concentration = min(
            max(
                target_concentration,
                self.beta_concentration_min + self.beta_eps,
            ),
            self.beta_concentration_max,
        )
        return self._softplus_inverse(
            target_concentration - self.beta_concentration_min
        )

    def default_head_bias_vector(
        self,
        *,
        log_std_init: float,
        log_std_init_route: float | None = None,
        log_std_init_speed: float | None = None,
        heading_amplitude1_std_init: float | None = None,
        heading_amplitude2_std_init: float | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if heading_amplitude1_std_init is None:
            heading_amplitude1_std_init = self.heading_amplitude1_std_init
        if heading_amplitude2_std_init is None:
            heading_amplitude2_std_init = self.heading_amplitude2_std_init

        base_bias = self.default_head_bias_from_log_std_init(log_std_init)
        bias = torch.full(
            (self.noise_pred_dim,),
            float(base_bias),
            device=device,
            dtype=dtype if dtype is not None else torch.float32,
        )
        for idx, group_name in enumerate(self.group_names):
            group_log_std_init = None
            if group_name == "route":
                group_log_std_init = log_std_init_route
            elif group_name == "target_speed":
                group_log_std_init = log_std_init_speed
            if group_log_std_init is not None:
                bias[idx] = self.default_head_bias_from_log_std_init(
                    float(group_log_std_init)
                )
        if self.use_spline_curvature_sampler:
            amp1 = max(float(heading_amplitude1_std_init), 1e-8)
            amp2 = max(float(heading_amplitude2_std_init), 1e-8)
            bias[self.base_noise_pred_dim] = math.log(amp1)
            bias[self.base_noise_pred_dim + 1] = math.log(amp2)
        return bias

    def diagnostics_from_head(
        self, mean_actions: torch.Tensor, head_pred: torch.Tensor
    ) -> NoiseDiagnostics:
        if self.distribution_type == "gaussian":
            return self.gaussian_diagnostics_from_head(head_pred)
        return self.beta_diagnostics_from_head(mean_actions, head_pred)

    def diagnostics_from_noise_pred(
        self, mean_actions: torch.Tensor, noise_pred: torch.Tensor
    ) -> NoiseDiagnostics:
        if self.distribution_type == "gaussian":
            return self.gaussian_diagnostics_from_noise_pred(noise_pred)
        return self.beta_diagnostics_from_noise_pred(mean_actions, noise_pred)

    def proba_distribution(
        self,
        mean_actions: torch.Tensor,
        noise_pred: torch.Tensor,
        *,
        from_head: bool = True,
        speed_logits: torch.Tensor | None = None,
    ) -> tuple[nn.Module | HybridActionDistribution, NoiseDiagnostics]:
        if mean_actions.ndim == 1:
            mean_actions = mean_actions.unsqueeze(0)
        if noise_pred.ndim == 1:
            noise_pred = noise_pred.unsqueeze(0)
        continuous_mean_actions = mean_actions[:, : self.continuous_action_dim]
        if from_head:
            diag = self.diagnostics_from_head(continuous_mean_actions, noise_pred)
        else:
            diag = self.diagnostics_from_noise_pred(continuous_mean_actions, noise_pred)
        dist: nn.Module | HybridActionDistribution | None = None
        if self.action_dist is not None and self.distribution_type == "gaussian":
            dist = self.action_dist.proba_distribution(
                continuous_mean_actions, diag.diag_log_std
            )
        elif self.action_dist is not None:
            dist = self.action_dist.proba_distribution(
                continuous_mean_actions, self.expand_group_values(diag.noise_pred)
            )
        if self.use_native_categorical_speed:
            if speed_logits is None:
                raise ValueError(
                    "native_categorical speed sampling requires target-speed logits."
                )
            dist = HybridActionDistribution(
                continuous_dist=dist,
                speed_logits=speed_logits,
                speed_values=self.speed_values,
                continuous_action_dim=self.continuous_action_dim,
            )
        if dist is None:
            raise RuntimeError("No action distribution constructed.")
        return dist, self._pad_speed_diagnostics(diag)

    def kl_divergence(
        self,
        old_dist: nn.Module | HybridActionDistribution,
        new_dist: nn.Module | HybridActionDistribution,
    ) -> torch.Tensor:
        if self.use_native_categorical_speed:
            assert isinstance(old_dist, HybridActionDistribution)
            assert isinstance(new_dist, HybridActionDistribution)
            return old_dist.kl_divergence(new_dist)
        kl_div = torch.distributions.kl_divergence(
            old_dist.distribution, new_dist.distribution
        )
        if kl_div.ndim > 1:
            kl_div = kl_div.sum(dim=1)
        return kl_div

    def _sample_spline_curvature_routes(
        self,
        route_norm: torch.Tensor,
        amp1_std: torch.Tensor,
        amp2_std: torch.Tensor,
    ) -> torch.Tensor:
        # route_norm shape: [B, N, 2]
        batch_size, num_points, _ = route_norm.shape
        if num_points < 2:
            return route_norm

        route_scale = self.action_codec.route_scale.to(
            device=route_norm.device, dtype=route_norm.dtype
        )
        route_m = route_norm * route_scale
        deltas = route_m[:, 1:, :] - route_m[:, :-1, :]
        ds = torch.linalg.norm(deltas, dim=-1)
        s_nodes = torch.cat(
            (
                torch.zeros(
                    (batch_size, 1), device=route_norm.device, dtype=route_norm.dtype
                ),
                torch.cumsum(ds, dim=1),
            ),
            dim=1,
        )
        total_len = torch.clamp(s_nodes[:, -1:], min=1e-6)
        u = s_nodes / total_len

        ramp_idx = min(2, num_points - 1)
        s_ramp_end = torch.clamp(s_nodes[:, ramp_idx : ramp_idx + 1], min=1e-6)
        z = torch.clamp(s_nodes / s_ramp_end, 0.0, 1.0)
        smooth = z * z * (3.0 - 2.0 * z)
        env = self.path_std_base_frac + (1.0 - self.path_std_base_frac) * smooth

        a1 = torch.randn(
            (batch_size, 1), device=route_norm.device, dtype=route_norm.dtype
        ) * amp1_std.view(batch_size, 1).to(route_norm.dtype)
        a2 = torch.randn(
            (batch_size, 1), device=route_norm.device, dtype=route_norm.dtype
        ) * amp2_std.view(batch_size, 1).to(route_norm.dtype)
        heading_perturb = env * (
            a1 * torch.sin(math.pi * u) + a2 * torch.sin(2.0 * math.pi * u)
        )

        seg_theta = torch.atan2(deltas[..., 1], deltas[..., 0])
        if num_points > 2:
            dtheta = seg_theta[:, 1:] - seg_theta[:, :-1]
            dtheta = torch.atan2(torch.sin(dtheta), torch.cos(dtheta))
            theta_unwrapped = torch.cat(
                (seg_theta[:, :1], seg_theta[:, :1] + torch.cumsum(dtheta, dim=1)),
                dim=1,
            )
        else:
            theta_unwrapped = seg_theta
        theta_nodes = torch.cat((theta_unwrapped[:, :1], theta_unwrapped), dim=1)
        theta_new = theta_nodes + heading_perturb

        x_new = torch.zeros_like(route_m[..., 0])
        y_new = torch.zeros_like(route_m[..., 1])
        x_new[:, 0] = route_m[:, 0, 0]
        y_new[:, 0] = route_m[:, 0, 1]
        for idx in range(1, num_points):
            ds_i = ds[:, idx - 1]
            theta_mid = 0.5 * (theta_new[:, idx - 1] + theta_new[:, idx])
            x_new[:, idx] = x_new[:, idx - 1] + torch.cos(theta_mid) * ds_i
            y_new[:, idx] = y_new[:, idx - 1] + torch.sin(theta_mid) * ds_i

        route_new_m = torch.stack((x_new, y_new), dim=-1)
        return route_new_m / route_scale

    def sample_actions(
        self,
        mean_actions: torch.Tensor,
        noise_diag: NoiseDiagnostics,
        dist: nn.Module,
        *,
        sample_type: str = "sample",
    ) -> torch.Tensor:
        if sample_type in ("mean", "mode", "deterministic"):
            return dist.mode()

        sampled = dist.sample()
        if not self.use_spline_curvature_sampler:
            return sampled
        if self.action_codec.slices.route is None:
            return sampled
        if noise_diag.path_amp1_std is None or noise_diag.path_amp2_std is None:
            return sampled

        route_slice = self.action_codec.slices.route
        route_mean = mean_actions[:, route_slice].view(
            -1, self.action_codec.num_route_points, 2
        )
        route_sampled = self._sample_spline_curvature_routes(
            route_mean,
            noise_diag.path_amp1_std,
            noise_diag.path_amp2_std,
        )
        sampled[:, route_slice] = route_sampled.reshape(sampled.shape[0], -1)
        return sampled
