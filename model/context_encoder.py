import torch
import torch.nn as nn


class LightweightContextEncoder(nn.Module):
    """
    iTransformer-inspired lightweight context encoder.

    - Input:  x_cat [B, L, D_plus_C]  (raw multivariate series + explicit context features concatenated)
    - Output: z_ctx [B, L, d_ctx]     (contextual / periodic representation)
    """

    def __init__(
        self,
        d_in: int,
        d_ctx: int = 32,
        nhead: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Project concatenated input to a small context hidden size.
        self.in_proj = nn.Linear(d_in, d_ctx)

        # Single-layer TransformerEncoder for lightweight contextual mixing.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_ctx,
            nhead=nhead,
            dim_feedforward=d_ctx * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,  # Keep tensors as [B, L, D]
            norm_first=True,   # Pre-norm for stability with shallow depth
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # Final normalization keeps output scale predictable.
        self.out_norm = nn.LayerNorm(d_ctx)

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_cat: [B, L, D_plus_C]

        Returns:
            z_ctx: [B, L, d_ctx]
        """
        # Shape: [B, L, d_ctx]
        h = self.in_proj(x_cat)

        # Shape: [B, L, d_ctx]
        h = self.encoder(h)

        # Shape: [B, L, d_ctx]
        z_ctx = self.out_norm(h)
        return z_ctx
