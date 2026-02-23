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


class ActionNoiseCodec:
    """Owns noise-head parameterization (grouped) and distribution construction."""

    def __init__(
        self,
        action_codec: ActionCodec,
        *,
        distribution_type: str = "gaussian",
        use_correlated_gaussian: bool = True,
        correlated_noise_rho: float = 0.8,
        noise_ramp: bool = True,
        log_std_min: float = -5.0,
        log_std_max: float = 1.0,
        beta_eps: float = 1e-4,
        beta_concentration_min: float = 2.0,
        beta_concentration_max: float = 200.0,
    ) -> None:
        self.action_codec = action_codec
        self.distribution_type = str(distribution_type)
        self.use_correlated_gaussian = bool(use_correlated_gaussian)
        self.correlated_noise_rho = float(correlated_noise_rho)
        self.noise_ramp = bool(noise_ramp)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
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

        if self.distribution_type == "gaussian":
            self.noise_pred_dim = len(self.group_names)
            if self.use_correlated_gaussian:
                self.action_dist: nn.Module = CorrelatedGaussianDistribution(
                    self.action_codec, rho=self.correlated_noise_rho
                )
            else:
                self.action_dist = DiagGaussianDistribution()
        elif self.distribution_type == "beta":
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

    def gaussian_diagnostics_from_head(
        self, head_pred: torch.Tensor
    ) -> NoiseDiagnostics:
        grouped_log_std = self._apply_group_noise_ramp(head_pred)
        diag_log_std = self.expand_group_values(grouped_log_std)
        diag_std = torch.exp(diag_log_std)
        return NoiseDiagnostics(
            noise_pred=grouped_log_std,
            diag_std=diag_std,
            diag_log_std=diag_log_std,
        )

    def gaussian_diagnostics_from_noise_pred(
        self, noise_pred: torch.Tensor
    ) -> NoiseDiagnostics:
        diag_log_std = self.expand_group_values(noise_pred)
        diag_std = torch.exp(diag_log_std)
        return NoiseDiagnostics(
            noise_pred=noise_pred,
            diag_std=diag_std,
            diag_log_std=diag_log_std,
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
