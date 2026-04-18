# #!/usr/bin/env python3
# """
# 多模态融合模型：IMU (CNN-LSTM-Attention) + 视觉 (MobileNetV3) 双流融合
# 架构：
#   IMU流：  Conv1D × 3 → BiLSTM → MultiheadAttention → 128-dim
#   视觉流：  MobileNetV3-Small (4帧平均池化) → 128-dim projection
#   融合：   Concat(128+128) → FC → num_classes
# """
#
# import torch
# import torch.nn as nn
# from torchvision.models import (
#     mobilenet_v3_small,
#     mobilenet_v3_large,
#     MobileNet_V3_Small_Weights,
#     MobileNet_V3_Large_Weights,
# )
#
#
# class IMUEncoder(nn.Module):
#     """
#     输入: (B, T, 6)  —— T=100时间步，6通道IMU
#     输出: (B, 128)
#     """
#
#     def __init__(self, input_dim: int = 6, cnn_channels: int = 128):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv1d(input_dim, 64, kernel_size=5, padding=2),
#             nn.BatchNorm1d(64),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(64, 128, kernel_size=5, padding=2),
#             nn.BatchNorm1d(128),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(128, cnn_channels, kernel_size=5, padding=2),
#             nn.BatchNorm1d(cnn_channels),
#             nn.ReLU(inplace=True),
#         )
#         self.lstm = nn.LSTM(
#             input_size=cnn_channels,
#             hidden_size=64,
#             num_layers=2,
#             batch_first=True,
#             bidirectional=True,  # → 128-dim
#         )
#         self.attn = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
#         self.out_dim = 128
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # x: (B, T, 6)
#         x = self.conv(x.transpose(1, 2))  # (B, cnn_channels, T)
#         x = x.transpose(1, 2)  # (B, T, cnn_channels)
#         x, _ = self.lstm(x)  # (B, T, 128)
#         x, _ = self.attn(x, x, x)  # (B, T, 128)
#         return x.mean(dim=1)  # (B, 128)
#
#
# class VisualEncoder(nn.Module):
#     """
#     输入: list of 4 tensors, each (B, 3, H, W)
#     输出: (B, 128)
#     """
#
#     def __init__(self, pretrained: bool = True, out_dim: int = 128):
#         super().__init__()
#         # weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
#         # backbone = mobilenet_v3_small(weights=weights)
#         weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
#         backbone = mobilenet_v3_large(weights=weights)
#         # 去掉最后的分类头，保留features+avgpool
#         self.features = backbone.features
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.proj = nn.Sequential(
#             nn.Linear(576, out_dim),
#             nn.ReLU(inplace=True),
#         )
#         self.out_dim = out_dim
#
#     def forward(self, frames: list[torch.Tensor]) -> torch.Tensor:
#         """
#         frames: list[Tensor(B,3,H,W)], len=4
#         returns: (B, out_dim)
#         """
#         feat_list = []
#         for frame in frames:
#             f = self.pool(self.features(frame)).flatten(1)  # (B, 576)
#             feat_list.append(f)
#         # 对4帧取平均 → 时序聚合
#         x = torch.stack(feat_list, dim=1).mean(dim=1)  # (B, 576)
#         return self.proj(x)  # (B, 128)
#
#
# class MultimodalFusion(nn.Module):
#     def __init__(
#         self,
#         num_classes: int = 5,
#         imu_input_dim: int = 6,
#         pretrained_visual: bool = True,
#         fusion_hidden: int = 128,
#         dropout: float = 0.4,
#     ):
#         super().__init__()
#         self.imu_enc = IMUEncoder(input_dim=imu_input_dim)
#         self.visual_enc = VisualEncoder(pretrained=pretrained_visual)
#
#         fusion_in = self.imu_enc.out_dim + self.visual_enc.out_dim  # 256
#
#         self.classifier = nn.Sequential(
#             nn.Linear(fusion_in, fusion_hidden),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),
#             nn.Linear(fusion_hidden, num_classes),
#         )
#
#     def forward(
#         self,
#         imu: torch.Tensor,  # (B, T, 6)
#         frames: list[torch.Tensor],  # list of 4 × (B, 3, H, W)
#     ) -> torch.Tensor:  # (B, num_classes)
#         imu_feat = self.imu_enc(imu)
#         visual_feat = self.visual_enc(frames)
#         fused = torch.cat([imu_feat, visual_feat], dim=1)
#         return self.classifier(fused)
#
#     def imu_only_forward(self, imu: torch.Tensor) -> torch.Tensor:
#         """仅使用IMU推理（视觉不可用时的降级模式）"""
#         imu_feat = self.imu_enc(imu)
#         # 用零向量占位视觉特征
#         vis_feat = torch.zeros(imu.size(0), self.visual_enc.out_dim, device=imu.device)
#         fused = torch.cat([imu_feat, vis_feat], dim=1)
#         return self.classifier(fused)
#
#
# if __name__ == "__main__":
#     # 快速验证
#     model = MultimodalFusion(num_classes=5)
#     imu = torch.randn(4, 100, 6)
#     frames = [torch.randn(4, 3, 224, 224) for _ in range(4)]
#     out = model(imu, frames)
#     print(f"Output shape: {out.shape}")
#     print(
#         f"IMU encoder params:    {sum(p.numel() for p in model.imu_enc.parameters()):,}"
#     )
#     print(
#         f"Visual encoder params: {sum(p.numel() for p in model.visual_enc.parameters()):,}"
#     )
#     print(f"Total params:          {sum(p.numel() for p in model.parameters()):,}")
#

#!/usr/bin/env python3
"""
更强的多模态融合模型：IMU + 视觉双流编码，配合时序注意力和门控融合。

相对旧版的主要增强：
1. IMU 分支升级为残差卷积 + BiLSTM + Transformer + attentive pooling
2. 视觉分支支持更强 backbone，并抽取单帧视觉特征
3. 融合头从单层 MLP 升级为门控融合 + 深层分类器
"""

from __future__ import annotations

import torch
import torch.nn as nn
from src.backbone import get_backbone


class ResidualConvBlock1D(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, stride: int = 1, dropout: float = 0.1
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=5,
                stride=stride,
                padding=2,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm1d(out_channels),
            )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x) + self.shortcut(x)
        return self.dropout(self.activation(out))


class AttentivePooling(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(x), dim=1)
        return torch.sum(x * weights, dim=1)


class IMUEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 22,
        base_channels: int = 96,
        lstm_hidden: int = 128,
        transformer_layers: int = 2,
        out_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(input_dim, base_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.GELU(),
        )
        self.conv_blocks = nn.Sequential(
            ResidualConvBlock1D(base_channels, base_channels, dropout=dropout),
            ResidualConvBlock1D(
                base_channels, base_channels * 2, stride=1, dropout=dropout
            ),
            ResidualConvBlock1D(base_channels * 2, base_channels * 2, dropout=dropout),
        )
        conv_out_dim = base_channels * 2
        self.lstm = nn.LSTM(
            input_size=conv_out_dim,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        seq_dim = lstm_hidden * 2
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=seq_dim,
            nhead=8,
            dim_feedforward=seq_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers
        )
        self.pool = AttentivePooling(seq_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(seq_dim),
            nn.Linear(seq_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.stem(x)
        x = self.conv_blocks(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = self.temporal_encoder(x)
        x = self.pool(x)
        return self.head(x)


class VisualEncoder(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        backbone_name: str = "mobilenet_v3_large",
        out_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.backbone, backbone_dim = get_backbone(backbone_name, pretrained)
        self.proj = nn.Sequential(
            nn.Linear(backbone_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out_dim = out_dim

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(frame)
        return self.proj(feat)


class MultimodalFusion(nn.Module):
    def __init__(
        self,
        num_classes: int = 6,
        imu_input_dim: int = 6,
        pretrained_visual: bool = True,
        visual_backbone: str = "mobilenet_v3_large",
        imu_hidden_dim: int = 256,
        visual_hidden_dim: int = 256,
        fusion_hidden: int = 256,
        dropout: float = 0.35,
    ):
        super().__init__()
        self.imu_enc = IMUEncoder(
            input_dim=imu_input_dim,
            out_dim=imu_hidden_dim,
            dropout=min(dropout, 0.25),
        )
        self.visual_enc = VisualEncoder(
            pretrained=pretrained_visual,
            backbone_name=visual_backbone,
            out_dim=visual_hidden_dim,
            dropout=min(dropout, 0.25),
        )

        self.imu_proj = nn.Sequential(
            nn.Linear(self.imu_enc.out_dim, fusion_hidden),
            nn.LayerNorm(fusion_hidden),
            nn.GELU(),
        )
        self.visual_proj = nn.Sequential(
            nn.Linear(self.visual_enc.out_dim, fusion_hidden),
            nn.LayerNorm(fusion_hidden),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(fusion_hidden * 2, fusion_hidden),
            nn.GELU(),
            nn.Linear(fusion_hidden, fusion_hidden),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(fusion_hidden * 4, fusion_hidden * 2),
            nn.LayerNorm(fusion_hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden * 2, fusion_hidden),
            nn.LayerNorm(fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, num_classes),
        )

    def _fuse_features(
        self, imu_feat: torch.Tensor, visual_feat: torch.Tensor
    ) -> torch.Tensor:
        imu_feat = self.imu_proj(imu_feat)
        visual_feat = self.visual_proj(visual_feat)
        gate = self.gate(torch.cat([imu_feat, visual_feat], dim=1))
        blended = gate * imu_feat + (1.0 - gate) * visual_feat
        fusion = torch.cat(
            [imu_feat, visual_feat, blended, torch.abs(imu_feat - visual_feat)],
            dim=1,
        )
        return self.classifier(fusion)

    def forward(self, imu: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
        imu_feat = self.imu_enc(imu)
        visual_feat = self.visual_enc(frame)
        return self._fuse_features(imu_feat, visual_feat)

    def imu_only_forward(self, imu: torch.Tensor) -> torch.Tensor:
        imu_feat = self.imu_enc(imu)
        visual_feat = torch.zeros(
            imu.size(0),
            self.visual_enc.out_dim,
            device=imu.device,
            dtype=imu.dtype,
        )
        return self._fuse_features(imu_feat, visual_feat)


def get_model(args):
    model = MultimodalFusion(
        args.num_classes,
        args.imu_input_dim,
        args.pretrained,
        args.backbone,
        args.imu_hidden_dim,
        args.visual_hidden_dim,
        args.fusion_hidden_dim,
    )
    return model
