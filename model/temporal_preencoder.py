"""
Lightweight iTransformer-inspired temporal pre-encoder.

Maps multivariate series X [B, L, D] to a compact per-timestep representation H [B, L, d_pre]
using a linear lift and one TransformerEncoder layer (temporal mixing). No explicit context,
no FFT, no auxiliary losses — for optional fusion upstream of MAAT.
"""

import torch
import torch.nn as nn


class LightweightTemporalPreEncoder(nn.Module):
    """Linear(D→d_pre) + 1× TransformerEncoder (batch_first), temporal attention over length L."""

    def __init__(
        self,
        d_in: int,
        d_pre: int = 32,
        nhead: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert d_pre % nhead == 0, "d_pre must be divisible by nhead for MultiheadAttention"

        # Project variables per timestep into model width
        self.input_proj = nn.Linear(d_in, d_pre)

        # Single encoder layer: light mixing along the time axis (batch_first keeps [B, L, d_pre])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_pre,
            nhead=nhead,
            dim_feedforward=d_pre * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # Stabilize scale before any downstream fusion
        self.out_norm = nn.LayerNorm(d_pre)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] raw multivariate window
        Returns:
            h: [B, L, d_pre] enhanced temporal representation
        """
        h = self.input_proj(x)
        h = self.encoder(h)
        h = self.out_norm(h)
        return h
