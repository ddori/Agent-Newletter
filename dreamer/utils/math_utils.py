"""DreamerV3 core math utilities: symlog, two-hot encoding, return normalization."""

import torch
import torch.nn.functional as F


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Symmetric logarithmic compression: sign(x) * ln(|x| + 1)."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse of symlog: sign(x) * (exp(|x|) - 1)."""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


def make_bins(num_bins: int = 255) -> torch.Tensor:
    """Create uniformly spaced bins in symlog space from -20 to 20."""
    return torch.linspace(-20.0, 20.0, num_bins)


def two_hot_encode(x: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Encode scalar values into two-hot vectors in symlog space.

    Args:
        x: Values to encode, any shape.
        bins: 1D tensor of bin centers.

    Returns:
        Two-hot encoded tensor with shape (*x.shape, num_bins).
    """
    x_symlog = symlog(x)
    x_symlog = x_symlog.unsqueeze(-1)
    bins = bins.to(x.device)

    # Clamp to bin range
    x_clamped = x_symlog.clamp(bins[0], bins[-1])

    # Find lower bin index
    below = (bins.unsqueeze(0) <= x_clamped).sum(dim=-1) - 1
    below = below.clamp(0, len(bins) - 2)
    above = below + 1

    # Interpolation weights
    below_val = bins[below]
    above_val = bins[above]
    weight_above = (x_clamped.squeeze(-1) - below_val) / (above_val - below_val + 1e-8)
    weight_above = weight_above.clamp(0.0, 1.0)
    weight_below = 1.0 - weight_above

    # Build two-hot
    result = torch.zeros(*x.shape, len(bins), device=x.device)
    result.scatter_(-1, below.unsqueeze(-1), weight_below.unsqueeze(-1))
    result.scatter_(-1, above.unsqueeze(-1), weight_above.unsqueeze(-1))
    return result


def two_hot_decode(probs: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Decode two-hot probabilities back to scalar values.

    Args:
        probs: Probability distribution over bins, shape (..., num_bins).
        bins: 1D tensor of bin centers.

    Returns:
        Decoded values in original (non-symlog) space.
    """
    bins = bins.to(probs.device)
    mean_symlog = (probs * bins).sum(dim=-1)
    return symexp(mean_symlog)


class ReturnNormalizer:
    """Percentile-based return normalization (DreamerV3 Section 3)."""

    def __init__(self, percentile_low: float = 5.0, percentile_high: float = 95.0,
                 decay: float = 0.99):
        self.low = percentile_low
        self.high = percentile_high
        self.decay = decay
        self._range_low = 0.0
        self._range_high = 1.0

    def update(self, returns: torch.Tensor) -> None:
        """Update running percentile estimates from a batch of returns."""
        low_pct = float(torch.quantile(returns.float().detach(), self.low / 100.0))
        high_pct = float(torch.quantile(returns.float().detach(), self.high / 100.0))
        self._range_low = self.decay * self._range_low + (1 - self.decay) * low_pct
        self._range_high = self.decay * self._range_high + (1 - self.decay) * high_pct

    def normalize(self, returns: torch.Tensor) -> torch.Tensor:
        """Normalize returns using percentile range, with max(1, range) denominator."""
        scale = max(1.0, self._range_high - self._range_low)
        return (returns - self._range_low) / scale
