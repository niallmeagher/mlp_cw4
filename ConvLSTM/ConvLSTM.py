import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a basic ConvLSTM cell


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        """
        Initialize a ConvLSTM cell.
        Args:
            input_dim: Number of channels in input tensor.
            hidden_dim: Number of channels in hidden state.
            kernel_size: Size of the convolutional kernel.
            bias: Whether or not to add the bias.
        """
        super(ConvLSTMCell, self).__init__()
        padding = kernel_size // 2  # to keep same spatial dimensions
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim,
                              kernel_size, padding=padding, bias=bias)

    def forward(self, x, h_prev, c_prev):
        """
        Forward pass for one time step.
        Args:
            x: input at current time step, shape (B, C, H, W)
            h_prev: previous hidden state, shape (B, hidden_dim, H, W)
            c_prev: previous cell state, shape (B, hidden_dim, H, W)
        Returns:
            h, c: updated hidden and cell states
        """
        combined = torch.cat([x, h_prev], dim=1)  # along channel dimension
        conv_out = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(conv_out, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)  # input gate
        f = torch.sigmoid(cc_f)  # forget gate
        o = torch.sigmoid(cc_o)  # output gate
        g = torch.tanh(cc_g)     # candidate memory
        c = f * c_prev + i * g   # new cell state
        h = o * torch.tanh(c)    # new hidden state
        return h, c
