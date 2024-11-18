import torch
import torch.nn as nn

class ECA(nn.Module):
    def __init__(self, in_channels, k_size=3):
        """
        ECA 模塊：使用一維卷積進行通道間交互
        :param in_channels: 輸入的通道數
        :param k_size: 1D卷積的核大小，默認為3
        """
        super(ECA, self).__init__()
        # 使用1D卷積進行通道間交互，padding='same' 保持張量大小不變
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k_size,
                                padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 全局平均池化後輸出為 (batch_size, channels, 1, 1)
        y = self.avg_pool(x)
        # Reshape 為 (batch_size, 1, channels) 以便於1D卷積操作
        y = self.conv1d(y.squeeze(-1).transpose(-1, -2))  # 維度轉置以符合 Conv1d 要求
        # 恢復到 (batch_size, channels, 1, 1)
        y = y.transpose(-1, -2).unsqueeze(-1)
        # 使用 Sigmoid 得到注意力權重並應用於輸入
        return x * self.sigmoid(y)