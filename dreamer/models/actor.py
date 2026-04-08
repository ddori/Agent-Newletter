"""Actor network for DreamerV3."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import numpy as np


class Actor(nn.Module):
    """Policy network: state -> continuous action distribution.

    Uses TruncatedNormal distribution for bounded continuous actions.
    """

    def __init__(self, state_dim: int, action_size: int,
                 mlp_units: int = 640, mlp_layers: int = 3,
                 init_std: float = 1.0, min_std: float = 0.1):
        super().__init__()
        self.action_size = action_size
        self._min_std = min_std

        layers = []
        dim = state_dim
        for _ in range(mlp_layers):
            layers.append(nn.Linear(dim, mlp_units))
            layers.append(nn.SiLU())
            dim = mlp_units
        self.trunk = nn.Sequential(*layers)

        self.mean_head = nn.Linear(mlp_units, action_size)
        self.std_head = nn.Linear(mlp_units, action_size)

        # Initialize for near-zero mean and reasonable std
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.zeros_(self.std_head.weight)
        nn.init.constant_(self.std_head.bias, np.log(np.exp(init_std) - 1.0))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Sample an action from the policy.

        Args:
            state: (..., state_dim) model state vector.

        Returns:
            (..., action_size) sampled action in [-1, 1].
        """
        dist = self.get_dist(state)
        action = dist.rsample()
        return torch.tanh(action)

    def get_dist(self, state: torch.Tensor) -> D.Normal:
        """Return the action distribution."""
        h = self.trunk(state)
        mean = self.mean_head(h)
        std = F.softplus(self.std_head(h)) + self._min_std
        return D.Normal(mean, std)

    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute log probability of action under the policy.

        Args:
            state: (..., state_dim)
            action: (..., action_size) tanh-squashed action.

        Returns:
            (...,) log probability.
        """
        dist = self.get_dist(state)
        # Inverse tanh to get pre-squash action
        raw_action = torch.atanh(action.clamp(-0.999, 0.999))
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        # Tanh correction
        log_prob -= torch.log(1.0 - action.pow(2) + 1e-6).sum(dim=-1)
        return log_prob

    def entropy(self, state: torch.Tensor) -> torch.Tensor:
        """Approximate entropy of the policy distribution."""
        dist = self.get_dist(state)
        return dist.entropy().sum(dim=-1)
