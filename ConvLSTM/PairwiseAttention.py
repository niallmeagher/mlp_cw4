import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a basic ConvLSTM cell


class PairwiseAttention(nn.Module):
    def __init__(self, in_channels, inter_channels=None):
        """
        Initializes a pairwise self-attention module.
        Args:
            in_channels: Number of channels in the input feature map.
            inter_channels: Number of channels in the intermediate feature space.
                            If None, defaults to half of in_channels.
        """
        super(PairwiseAttention, self).__init__()
        if inter_channels is None:
            inter_channels = max(1, in_channels // 2)
        self.conv_q = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.conv_k = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.conv_v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        Forward pass for pairwise self-attention.
        Args:
            x: Input feature map of shape (B, C, H, W)
        Returns:
            out: Feature map after applying self-attention (same shape as x)
        """
        B, C, H, W = x.size()
        N = H * W
        q = self.conv_q(x).view(B, -1, N)      # (B, inter_channels, N)
        k = self.conv_k(x).view(B, -1, N)        # (B, inter_channels, N)
        v = self.conv_v(x).view(B, -1, N)        # (B, C, N)
        q = q.permute(0, 2, 1)                   # (B, N, inter_channels)
        attn = torch.bmm(q, k)                   # (B, N, N)
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(v, attn.permute(0, 2, 1))  # (B, C, N)
        out = out.view(B, C, H, W)
        out = self.gamma * out + x
        return out
