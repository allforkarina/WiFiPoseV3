from __future__ import annotations

import torch
from torch import nn, Tensor


class TemporalResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.block(x) + x)


class MultiScaleTemporalPoseTCN(nn.Module):
    """Frame encoder + multi-scale dilated temporal conv for multi-frame AoA pose regression."""

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
        self.num_joints = num_joints
        self.out_dim = out_dim
        self.feature_dim = hidden_dim

        frame_dim = max(128, hidden_dim // 2)
        self.frame_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, frame_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(frame_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.temporal_proj = nn.Conv1d(frame_dim, hidden_dim, kernel_size=1)
        self.temporal_blocks = nn.ModuleList(
            [
                TemporalResidualBlock(hidden_dim, kernel_size=3, dilation=1, dropout=dropout),
                TemporalResidualBlock(hidden_dim, kernel_size=3, dilation=2, dropout=dropout),
                TemporalResidualBlock(hidden_dim, kernel_size=5, dilation=1, dropout=dropout),
                TemporalResidualBlock(hidden_dim, kernel_size=5, dilation=2, dropout=dropout),
            ]
        )
        self.fusion = nn.Sequential(
            nn.Conv1d(hidden_dim * len(self.temporal_blocks), hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_joints * out_dim),
        )

    def forward_features(self, x: Tensor) -> Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1).unsqueeze(1)
        elif x.dim() == 3:
            x = x.unsqueeze(2)
        if x.dim() != 4:
            raise ValueError(f"Expected input with shape (B,T,L) or (B,T,1,L), got {tuple(x.shape)}")

        batch_size, time_steps, _, input_length = x.shape
        frames = x.view(batch_size * time_steps, 1, input_length)
        frame_feats = self.frame_encoder(frames).squeeze(-1)
        frame_feats = frame_feats.view(batch_size, time_steps, -1).transpose(1, 2)

        temporal = self.temporal_proj(frame_feats)
        multi_scale = [block(temporal) for block in self.temporal_blocks]
        fused = self.fusion(torch.cat(multi_scale, dim=1))
        center_idx = fused.size(-1) // 2
        return fused[:, :, center_idx]

    def forward_head(self, feats: Tensor) -> Tensor:
        out = self.head(feats)
        return out.view(out.size(0), self.num_joints, self.out_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_head(self.forward_features(x))


if __name__ == "__main__":
    model = MultiScaleTemporalPoseTCN()
    y = model(torch.randn(2, 5, 181))
    print(y.shape)
