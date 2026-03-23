from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn, Tensor


class ConvBaseline(nn.Module):
    """Lightweight 1D-CNN baseline for AoA -> 2D pose.

    Expected input: X of shape (B, 1, 181)
    Output: pose of shape (B, 17, 2)
    """

    def __init__(
        self,
        input_channels: int = 1,
        input_length: int = 181,
        hidden_dim: int = 256,
        num_joints: int = 17,
        out_dim: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.input_channels = input_channels
        self.input_length = input_length
        self.num_joints = num_joints
        self.out_dim = out_dim
        self.feature_dim = hidden_dim

        # A small stack of 1D conv + BN + ReLU layers
        self.feature = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(128, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool1d(1),  # -> (B, hidden_dim, 1)
        )

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, num_joints * out_dim)

    def forward_features(self, x: Tensor) -> Tensor:
        # Ensure expected shape (B, C, L)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        feats = self.feature(x).squeeze(-1)  # (B, hidden_dim)
        return self.dropout(feats)

    def forward_head(self, feats: Tensor) -> Tensor:
        out = self.head(feats)  # (B, num_joints * out_dim)
        pose = out.view(out.size(0), self.num_joints, self.out_dim)
        return pose

    def forward(self, x: Tensor) -> Tensor | Tuple[Tensor, Optional[Tensor]]:
        """Forward pass.

        Args:
            x: Tensor of shape (B, C=1, L=181).

        Returns:
            pose: Tensor of shape (B, 17, 2).
            Optionally, a confidence tensor could be added later; for now
            only the pose tensor is returned to match the minimal interface.
        """
        return self.forward_head(self.forward_features(x))


if __name__ == "__main__":  # simple smoke test
    m = ConvBaseline()
    dummy = torch.randn(2, 1, 181)
    y = m(dummy)
    print(y.shape)
