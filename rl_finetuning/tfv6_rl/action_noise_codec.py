from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Beta, MultivariateNormal, Normal

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


class ActionNoiseCodec:
    """Owns noise-head parameterization (grouped) and distribution construction."""

    SAMPLER_SPLINE_CURVATURE = "spline_curvature_perturbation"
    SAMPLER_LEGACY = "legacy_noise_sampling"

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
        beta_eps: float = 1e-4,
        beta_concentration_min: float = 2.0,
        beta_concentration_max: float = 200.0,
    ) -> None:
        self.action_codec = action_codec
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
        self.beta_eps = float(beta_eps)
        self.beta_concentration_min = float(beta_concentration_min)
        self.beta_concentration_max = float(beta_concentration_max)

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
        if not self.group_names:
            raise RuntimeError("No active action heads found for ActionNoiseCodec.")

        if self.sampling_technique not in (
            self.SAMPLER_SPLINE_CURVATURE,
            self.SAMPLER_LEGACY,
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

        self.base_noise_pred_dim = len(self.group_names)
        self.route_group_index = (
            self.group_names.index("route") if "route" in self.group_names else None
        )
        self.use_spline_curvature_sampler = (
            self.distribution_type == "gaussian"
            and self.sampling_technique == self.SAMPLER_SPLINE_CURVATURE
            and self.route_group_index is not None
        )

        if self.distribution_type == "gaussian":
            self.noise_pred_dim = self.base_noise_pred_dim + (
                2 if self.use_spline_curvature_sampler else 0
            )
            if self.use_correlated_gaussian:
                self.action_dist: nn.Module = CorrelatedGaussianDistribution(
                    self.action_codec, rho=self.correlated_noise_rho
                )
            else:
                self.action_dist = DiagGaussianDistribution()
        elif self.distribution_type == "beta":
            if self.sampling_technique != self.SAMPLER_LEGACY:
                raise ValueError(
                    "spline_curvature_perturbation currently supports only gaussian action noise."
                )
            self.noise_pred_dim = len(
                self.group_names
            )  # one concentration per active head type
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
    ) -> tuple[nn.Module, NoiseDiagnostics]:
        if mean_actions.ndim == 1:
            mean_actions = mean_actions.unsqueeze(0)
        if noise_pred.ndim == 1:
            noise_pred = noise_pred.unsqueeze(0)
        if from_head:
            diag = self.diagnostics_from_head(mean_actions, noise_pred)
        else:
            diag = self.diagnostics_from_noise_pred(mean_actions, noise_pred)
        if self.distribution_type == "gaussian":
            dist = self.action_dist.proba_distribution(mean_actions, diag.diag_log_std)
        else:
            dist = self.action_dist.proba_distribution(
                mean_actions, self.expand_group_values(diag.noise_pred)
            )
        return dist, diag

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
