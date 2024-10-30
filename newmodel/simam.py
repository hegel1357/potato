import torch
import torch.nn as nn

class SimAM(nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        return f"{self.__class__.__name__}(lambda={self.e_lambda})"

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        # 獲取輸入張量的批次大小、通道數、高度和寬度
        b, c, h, w = x.size()
        
        # 計算特徵圖空間維度減去1的值
        n = w * h - 1

        # 計算每個位置與均值的平方差
        x_minus_mean_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        
        # 計算注意力權重
        y = x_minus_mean_square / (4 * (x_minus_mean_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        # 將注意力權重應用於輸入張量並返回結果
        return x * self.activation(y)