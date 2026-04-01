from __future__ import annotations

import torch
from torch import nn, Tensor

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

def grad_reverse(x, alpha=1.0):
    return GradientReversalLayer.apply(x, alpha)


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        return self.act(out)


class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

def grad_reverse(x, alpha=1.0):
    return GradientReversalLayer.apply(x, alpha)


class ResNet1DPose(nn.Module):
    """Residual 1D CNN for AoA -> normalized 2D pose."""

    def __init__(
        self,
        input_channels: int = 1,
        input_length: int = 181,
        hidden_dim: int = 256,
        num_joints: int = 17,
        out_dim: int = 2,
        dropout: float = 0.2,
        num_envs: int = 0, # Phase 1.3: DANN Classifier
    ) -> None:
        super().__init__()
        self.num_joints = num_joints
        self.out_dim = out_dim
        self.feature_dim = hidden_dim
        self.num_envs = num_envs

        base_dim = max(64, hidden_dim // 4)
        mid_dim = max(128, hidden_dim // 2)

        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, base_dim, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(base_dim),
            nn.ReLU(inplace=True),
        )
        self.backbone = nn.Sequential(
            ResidualBlock1D(base_dim, base_dim, stride=1, dropout=dropout * 0.5),
            ResidualBlock1D(base_dim, mid_dim, stride=2, dropout=dropout * 0.5),
            ResidualBlock1D(mid_dim, hidden_dim, stride=2, dropout=dropout),
            ResidualBlock1D(hidden_dim, hidden_dim, stride=1, dropout=dropout),
        )
        
        # Pass a dummy tensor to correctly calculate the flattened size after dynamic temporal decimation
        dummy_in = torch.zeros(1, input_channels, input_length)
        with torch.no_grad():
            dummy_stem = self.stem(dummy_in)
            dummy_backbone = self.backbone(dummy_stem)
            flat_size = dummy_backbone.view(1, -1).size(1)

        self.feature_dim = flat_size
        
        self.head = nn.Sequential(
            nn.Flatten(1),
            nn.Dropout(dropout),
            nn.Linear(flat_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_joints * out_dim),
        )

        # Phase 1.3: Environment Classifier for DANN
        if num_envs > 0:
            self.env_classifier = nn.Sequential(
                nn.Flatten(1),
                nn.Linear(flat_size, hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim // 2, num_envs)
            )

    def forward_features(self, x: Tensor) -> Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        feats = self.stem(x)
        feats = self.backbone(feats)
        return torch.flatten(feats, 1)

    def forward_head(self, feats: Tensor) -> Tensor:
        out = self.head[1:](self.head[0](feats))
        return out.view(out.size(0), self.num_joints, self.out_dim)

    def forward_env(self, feats: Tensor, alpha: float = 1.0) -> Tensor:
        if self.num_envs <= 0:
            raise RuntimeError("Environment classifier not initialized (num_envs <= 0)")
        reversed_feats = grad_reverse(feats, alpha)
        return self.env_classifier(reversed_feats)

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_head(self.forward_features(x))


if __name__ == "__main__":
    model = ResNet1DPose()
    y = model(torch.randn(2, 1, 181))
    print(y.shape)
