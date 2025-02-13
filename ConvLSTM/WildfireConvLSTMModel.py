import torch
import torch.nn as nn
import torch.nn.functional as F
from ConvLSTM import ConvLSTMCell
from PairwiseAttention import PairwiseAttention


class WildfireConvLSTMModel(nn.Module):
    def __init__(self, input_channels, hidden_dims, kernel_size=3, time_steps=10):
        """
        Args:
            input_channels: number of input channels (e.g. fire predictors)
            hidden_dims: list of hidden dims for each ConvLSTM layer
            kernel_size: kernel size for conv operations
            time_steps: number of input time steps (e.g., 10)
        """
        super(WildfireConvLSTMModel, self).__init__()
        self.time_steps = time_steps
        self.cell1 = ConvLSTMCell(input_channels, hidden_dims[0], kernel_size)
        self.cell2 = ConvLSTMCell(hidden_dims[0], hidden_dims[1], kernel_size)
        self.attention = PairwiseAttention(hidden_dims[1])
        # For supervised training, T == self.time_steps
        self.conv3d = nn.Conv3d(
            hidden_dims[1], 256, kernel_size=(time_steps, 1, 1))
        self.conv_final = nn.Conv2d(256, 1, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, T, C, H, W) with T == self.time_steps.
        Returns:
            Fire probability map: (B, 1, H, W)
        """
        B, T, C, H, W = x.size()
        h1 = torch.zeros(B, self.cell1.hidden_dim, H, W, device=x.device)
        c1 = torch.zeros(B, self.cell1.hidden_dim, H, W, device=x.device)
        h2 = torch.zeros(B, self.cell2.hidden_dim, H, W, device=x.device)
        c2 = torch.zeros(B, self.cell2.hidden_dim, H, W, device=x.device)
        outputs = []
        for t in range(T):
            x_t = x[:, t, :, :, :]
            h1, c1 = self.cell1(x_t, h1, c1)
            h2, c2 = self.cell2(h1, h2, c2)
            outputs.append(h2)
        outputs = torch.stack(outputs, dim=1)  # (B, T, hidden_dim, H, W)
        outputs_att = []
        for t in range(T):
            out_t = outputs[:, t, :, :, :]
            out_t = self.attention(out_t)
            outputs_att.append(out_t)
        outputs_att = torch.stack(outputs_att, dim=1)
        outputs_att = outputs_att.permute(
            0, 2, 1, 3, 4)  # (B, hidden_dim, T, H, W)
        x3d = self.conv3d(outputs_att)  # (B, 256, 1, H, W)
        x3d = x3d.squeeze(2)            # (B, 256, H, W)
        output = self.conv_final(x3d)   # (B, 1, H, W)
        return output

    def forward_rl(self, x):
        """
        Helper for RL fine-tuning.
        For RL, we assume state sₜ has shape (B, T, C, H, W) with T == self.time_steps.
        Returns a scalar per sample (mean over spatial dims) representing the policy's output.
        """
        out = self.forward(x)  # (B, 1, H, W)
        out_scalar = out.view(out.size(0), -1).mean(dim=1, keepdim=True)
        return out_scalar
