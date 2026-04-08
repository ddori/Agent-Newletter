"""CNN image decoder for DreamerV3."""

import torch
import torch.nn as nn


class Decoder(nn.Module):
    """Transpose-CNN decoder: (B, state_dim) -> (B, 3, 64, 64).

    Mirrors the encoder architecture with transposed convolutions.
    """

    def __init__(self, state_dim: int, depth: int = 48, layers: int = 4,
                 kernels: tuple = (4, 4, 4, 4)):
        super().__init__()
        assert len(kernels) == layers

        # Compute the spatial size at the bottleneck
        # With 4 layers of stride-2 on 64x64: 64 -> 32 -> 16 -> 8 -> 4
        self._bottleneck_channels = depth * (2 ** (layers - 1))
        self._spatial = 64 // (2 ** layers)  # 4 for default

        self.linear = nn.Linear(state_dim, self._bottleneck_channels * self._spatial ** 2)

        deconvs = []
        ch_in = self._bottleneck_channels
        for i in range(layers - 1, -1, -1):
            ch_out = depth * (2 ** (i - 1)) if i > 0 else 3
            if i > 0:
                deconvs.append(nn.ConvTranspose2d(ch_in, ch_out, kernels[layers - 1 - i],
                                                   stride=2, padding=1, output_padding=0))
                deconvs.append(nn.SiLU())
            else:
                deconvs.append(nn.ConvTranspose2d(ch_in, ch_out, kernels[layers - 1 - i],
                                                   stride=2, padding=1, output_padding=0))
            ch_in = ch_out
        self.deconvs = nn.Sequential(*deconvs)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Decode model state to image reconstruction.

        Args:
            state: (B, state_dim) concatenated deterministic + stochastic state.

        Returns:
            (B, 3, 64, 64) reconstructed image (logits, not sigmoid).
        """
        h = self.linear(state)
        h = h.reshape(-1, self._bottleneck_channels, self._spatial, self._spatial)
        return self.deconvs(h)
