import torch
import torch.nn as nn

class ShuffleAttention(nn.Module):
    def __init__(self, input_channels: int, squeeze_channels: int, groups: int = 8):
        """
        初始化 ShuffleAttention 模塊。
        
        Args:
            input_channels (int): 輸入通道數。
            squeeze_channels (int): 壓縮通道數。
            groups (int): 分組數量，用於通道隨機混排。
        """
        super().__init__()
        self.groups = groups
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, kernel_size=1)  # 1x1 卷積
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, kernel_size=1)  # 1x1 卷積
        self.sigmoid = nn.Sigmoid()  # Sigmoid 激活函數

    def channel_shuffle(self, x):
        """進行通道隨機混排"""
        b, c, h, w = x.size()
        # 將特徵圖分成 groups 個組，然後進行混排
        x = x.view(b, self.groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)  # 通道隨機混排
        return x.contiguous().view(b, -1, h, w)  # 重新調整形狀

    def forward(self, x):
        b, c, h, w = x.size()
        # 先通過全局平均池化
        avg_out = self.avgpool(x)  # (b, c, 1, 1)
        avg_out = self.fc1(avg_out)  # (b, squeeze_channels, 1, 1)
        avg_out = nn.ReLU()(avg_out)  # 激活
        avg_out = self.fc2(avg_out)  # (b, c, 1, 1)
        scale = self.sigmoid(avg_out)  # (b, c, 1, 1)

        # 應用注意力
        x = x * scale  # (b, c, h, w)

        # 通道隨機混排
        x = self.channel_shuffle(x)  # (b, c, h, w)

        return x
