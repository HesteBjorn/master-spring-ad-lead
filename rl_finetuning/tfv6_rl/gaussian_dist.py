from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal


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
