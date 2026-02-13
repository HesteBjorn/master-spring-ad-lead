from __future__ import annotations

import torch
from torch import nn
from torch.distributions import MultivariateNormal, Normal

from rl_finetuning.tfv6_rl.action_codec import ActionCodec


def sum_independent_dims(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim > 1:
        return tensor.sum(dim=1)
    return tensor.sum()


class DiagGaussianDistribution(nn.Module):
    """Simple diagonal Gaussian distribution wrapper for PPO."""

    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.distribution: Normal | None = None
        self.log_std_min = -20.0
        self.log_std_max = 2.0

    def proba_distribution(
        self, mean_actions: torch.Tensor, log_std: torch.Tensor
    ) -> DiagGaussianDistribution:
        if log_std.ndim == 1:
            log_std = log_std.unsqueeze(0).expand_as(mean_actions)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        action_std = torch.exp(log_std)
        self.distribution = Normal(mean_actions, action_std)
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        assert self.distribution is not None
        log_prob = self.distribution.log_prob(actions)
        return sum_independent_dims(log_prob)

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
        # Not used in this setup; return zero to avoid crashes if called.
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
        if mean_actions.ndim == 1:
            mean_actions = mean_actions.unsqueeze(0)
        if log_std.ndim == 1:
            log_std = log_std.unsqueeze(0).expand_as(mean_actions)

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
                speed_cov = (
                    speed_std.pow(2).reshape(1, 1)
                    + torch.eye(1, device=device, dtype=dtype) * self.jitter
                )
                blocks.append(speed_cov)

            cov = blocks[0] if len(blocks) == 1 else torch.block_diag(*blocks)
            scale_trils.append(torch.linalg.cholesky(cov))

        scale_tril = torch.stack(scale_trils, dim=0)
        self.distribution = MultivariateNormal(loc=mean_actions, scale_tril=scale_tril)
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
