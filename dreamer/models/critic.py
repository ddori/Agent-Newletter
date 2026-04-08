"""Critic network with EMA target for DreamerV3."""

import torch
import torch.nn as nn

from dreamer.models.heads import SymlogTwoHotHead


class Critic(nn.Module):
    """Value network with slow EMA target network.

    Predicts state values using symlog two-hot distribution.
    """

    def __init__(self, state_dim: int, mlp_units: int = 640, mlp_layers: int = 3,
                 num_bins: int = 255, ema_decay: float = 0.98):
        super().__init__()
        self.ema_decay = ema_decay

        # Main critic
        self.critic = SymlogTwoHotHead(
            in_dim=state_dim,
            mlp_units=mlp_units,
            mlp_layers=mlp_layers,
            num_bins=num_bins,
        )

        # EMA target critic (not updated by gradient)
        self.target = SymlogTwoHotHead(
            in_dim=state_dim,
            mlp_units=mlp_units,
            mlp_layers=mlp_layers,
            num_bins=num_bins,
        )
        self.target.load_state_dict(self.critic.state_dict())
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Predict value using main critic."""
        return self.critic.predict(state)

    def target_value(self, state: torch.Tensor) -> torch.Tensor:
        """Predict value using EMA target critic."""
        with torch.no_grad():
            return self.target.predict(state)

    def loss(self, state: torch.Tensor, target_value: torch.Tensor) -> torch.Tensor:
        """Compute critic loss against lambda-return targets.

        Args:
            state: (..., state_dim) model states.
            target_value: (...) target values (lambda returns).

        Returns:
            Scalar loss.
        """
        return self.critic.loss(state, target_value)

    def update_target(self) -> None:
        """Update EMA target network."""
        with torch.no_grad():
            for param, target_param in zip(self.critic.parameters(),
                                            self.target.parameters()):
                target_param.data.mul_(self.ema_decay).add_(
                    param.data, alpha=1.0 - self.ema_decay
                )
