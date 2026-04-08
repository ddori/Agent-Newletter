"""CNN image encoder for DreamerV3."""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Multi-layer CNN encoder: (B, 3, 64, 64) -> (B, embed_dim).

    DreamerV3 default: 4 layers, depth multiplier 48, kernel 4, stride 2, SiLU + LayerNorm.
    """

    def __init__(self, in_channels: int = 3, depth: int = 48, layers: int = 4,
                 kernels: tuple = (4, 4, 4, 4)):
        super().__init__()
        assert len(kernels) == layers

        convs = []
        ch_in = in_channels
        for i in range(layers):
            ch_out = depth * (2 ** i)
            convs.append(nn.Conv2d(ch_in, ch_out, kernels[i], stride=2, padding=1))
            convs.append(nn.SiLU())
            ch_in = ch_out
        self.convs = nn.Sequential(*convs)

        # Compute output size by doing a forward pass with dummy input
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 64, 64)
            out = self.convs(dummy)
            self.embed_dim = int(out.reshape(1, -1).shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image observation.

        Args:
            x: (B, 3, 64, 64) float tensor in [0, 1].

        Returns:
            (B, embed_dim) embedding.
        """
        h = self.convs(x)
        return h.reshape(h.shape[0], -1)
