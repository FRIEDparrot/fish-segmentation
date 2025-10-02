from torch import nn
import torch.nn.functional as F
import torch


class FishSegmentClassifier(nn.Module):
    """
    A neural network module for fish segmentation classification.

    This module is designed to classify input feature maps by predicting class
    probabilities for each input. It uses an adaptive average pooling layer to
    reduce spatial dimensions, followed by a fully connected layer for classification.

    Attributes:
        pool (nn.AdaptiveAvgPool2d): Adaptive average pooling layer that reduces the spatial dimensions to 1x1.
        fc (nn.Linear): Fully connected layer for classification, including an additional output for the background class.
    """

    def __init__(self, in_ch=2048, num_classes=9):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_ch, num_classes + 1)  # +1 for background shift

    def forward(self, x):
        return self.fc(self.pool(x).flatten(1))


class FishSegmentBBoxHead(nn.Module):
    """
    FishSegmentBBoxHead predicts normalized bounding box coordinates from input feature maps using attention and MLP layers.

    Input shape:
        x: torch.Tensor of shape (B, C, H, W)
            B: batch size
            C: number of channels (default: 256)
            H, W: spatial dimensions
    Output shape:
        torch.Tensor of shape (B, 4)
            Each row contains normalized bounding box coordinates (e.g., [x_min, y_min, x_max, y_max]) for each input in the batch.

    Attributes:
        attn (nn.MultiheadAttention): Multi-head self-attention layer operating on channel dimension.
        mlp (nn.Sequential): MLP for regressing bounding box coordinates from pooled features.
    """

    def __init__(self, in_ch=256):
        super().__init__()
        self.attn = nn.MultiheadAttention(in_ch, num_heads=8, batch_first=False)
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_ch),
            nn.Linear(in_ch, in_ch),
            nn.ReLU(),
            nn.Linear(in_ch, 4)
        )

    def forward(self, x):
        """
        Forward pass for bounding box prediction.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
        Returns:
            torch.Tensor: Output tensor of shape (B, 4) with normalized bounding box coordinates.
        """
        B, C, H, W = x.shape
        seq = x.view(B, C, H * W).permute(2, 0, 1)  # (L,B,C)
        attn_out, _ = self.attn(seq, seq, seq)
        pooled = attn_out.mean(0)  # (B,C)
        return self.mlp(pooled)  # (B,4) normalized bbox


class FishSegmentationHead(nn.Module):
    """
    FishSegmentationHead predicts pixel-wise segmentation masks from input feature maps using attention and convolutional decoding layers.

    Input shape:
        x: torch.Tensor of shape (B, C, H, W)
            B: batch size
            C: number of channels (default: 256)
            H, W: spatial dimensions
    Output shape:
        torch.Tensor of shape (B, num_classes + 1, H, W)
            Each output contains per-pixel logits for each class (including background),
            upsampled to (SEG_H, SEG_W) resolution (default: 512x512).

    Attributes:
        reduce (nn.Conv2d): 1x1 convolution to reduce channel dimension.
        attn (nn.MultiheadAttention): Multi-head self-attention layer operating on reduced features.
        decode (nn.Sequential): Convolutional decoder for segmentation logits.
    """

    def __init__(self, in_ch=256, hidden=128, num_classes=9):
        super().__init__()
        self.reduce = nn.Conv2d(in_ch, hidden, 1)
        self.attn = nn.MultiheadAttention(embed_dim=hidden, num_heads=8, batch_first=False)
        self.decode = nn.Sequential(
            nn.Conv2d(hidden, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, num_classes + 1, 1)
        )

    def forward(self, x, seg_output_size: tuple):
        """
        Forward pass for segmentation mask prediction.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            seg_output_size (tuple): Desired output size (H, W) for segmentation masks. (we may need down sampling)
        Returns:
            torch.Tensor: Output tensor of shape (B, num_classes + 1, H, W) with per-pixel logits.
        """
        B, C, H, W = x.shape
        x = self.reduce(x)  # (B,hidden,H,W)
        seq = x.view(B, -1, H * W).permute(2, 0, 1)  # (L,B,hidden)
        attn_out, _ = self.attn(seq, seq, seq)
        attn_out = attn_out.permute(1, 2, 0).view(B, -1, H, W)
        logits = self.decode(attn_out)
        logits = F.interpolate(logits, size=seg_output_size, mode='bilinear', align_corners=False)
        return logits
