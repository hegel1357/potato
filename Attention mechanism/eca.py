import torch
import torch.nn as nn
import torch.nn.functional as F

class EfficientChannelAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super(EfficientChannelAttention, self).__init__()
        self.kernel_size = kernel_size
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=self.kernel_size, padding=self.kernel_size // 2, groups=in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channel, _, _ = x.size()
        # Squeeze operation
        y = self.avg_pool(x).view(batch_size, channel)
        # Use 1D convolution for channel attention
        y = y.view(batch_size, channel, 1, 1)
        y = self.conv(y)
        # Activation
        y = self.sigmoid(y)
        # Scale the input
        return x * y

# Example usage
# eca = EfficientChannelAttention(in_channels=64)
# output = eca(input_tensor)