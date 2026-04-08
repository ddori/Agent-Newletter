"""Prediction heads for DreamerV3: reward and continue predictors."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from dreamer.utils.math_utils import make_bins, two_hot_encode, two_hot_decode, symlog


class SymlogTwoHotHead(nn.Module):
    """MLP head that predicts a scalar via symlog two-hot distribution.

    Used for reward prediction and critic value prediction in DreamerV3.
    """

    def __init__(self, in_dim: int, mlp_units: int = 640, mlp_layers: int = 3,
                 num_bins: int = 255):
        super().__init__()
        self.num_bins = num_bins
        self.register_buffer("bins", make_bins(num_bins))

        layers = []
        dim = in_dim
        for _ in range(mlp_layers):
            layers.append(nn.Linear(dim, mlp_units))
            layers.append(nn.SiLU())
            dim = mlp_units
        layers.append(nn.Linear(mlp_units, num_bins))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits over bins, shape (..., num_bins)."""
        return self.net(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted scalar value (decoded from softmax over bins)."""
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        return two_hot_decode(probs, self.bins)

    def loss(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss against two-hot encoded target.

        Args:
            x: Input features, shape (..., in_dim).
            target: Target scalar values, shape (...).

        Returns:
            Scalar loss.
        """
        logits = self.forward(x)
        target_encoded = two_hot_encode(target, self.bins)
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(target_encoded * log_probs).sum(dim=-1)
        return loss.mean()


class ContinueHead(nn.Module):
    """MLP head that predicts episode continuation probability (Bernoulli)."""

    def __init__(self, in_dim: int, mlp_units: int = 640, mlp_layers: int = 3):
        super().__init__()
        layers = []
        dim = in_dim
        for _ in range(mlp_layers):
            layers.append(nn.Linear(dim, mlp_units))
            layers.append(nn.SiLU())
            dim = mlp_units
        layers.append(nn.Linear(mlp_units, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return continuation logit, shape (..., 1)."""
        return self.net(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return continuation probability."""
        return torch.sigmoid(self.forward(x)).squeeze(-1)

    def loss(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Binary cross-entropy loss for continuation prediction.

        Args:
            x: Input features.
            target: 1.0 for continuing, 0.0 for terminal.

        Returns:
            Scalar loss.
        """
        logits = self.forward(x).squeeze(-1)
        return F.binary_cross_entropy_with_logits(logits, target)
