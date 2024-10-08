import torch
import torch.nn as nn

class FCA(nn.Module):
    def __init__(self, in_planes, reduction_ratio=16):
        super(FCA, self).__init__()
        # 使用1x1卷積代替全連接層來進行通道間的交互
        self.conv = nn.Conv2d(in_planes, in_planes // reduction_ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_planes // reduction_ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 全局平均池化
        avg_out = torch.mean(x, dim=[2, 3], keepdim=True)  # Global Avg Pooling
        # 兩層卷積和激活
        out = self.conv(avg_out)
        out = self.relu(out)
        out = self.conv2(out)
        # 使用Sigmoid激活得到注意力權重
        return x * self.sigmoid(out)
