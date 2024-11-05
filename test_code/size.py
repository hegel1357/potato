import torch
from torchsummary import summary
from torchvision.models import regnet_y_400mf  # 明確匯入模型
from fvcore.nn import FlopCountAnalysis, parameter_count
import torch.nn as nn

# 定義一個函式來計算參數和 FLOPs


def calculate_model_complexity(model, input_size, device):
    model.to(device)

    # 打印參數量
    print("模型參數量：")
    summary(model, input_size=input_size)

    # 計算 FLOPs
    input_tensor = torch.randn(1, *input_size).to(device)
    flops = FlopCountAnalysis(model, input_tensor)
    params = parameter_count(model)

    print(f"FLOPs: {flops.total() / 1e9:.2f} GFLOPs")
    print(f"參數量: {params[''] / 1e6:.2f} M")


# 設置模型和裝置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = regnet_y_400mf(weights=None)  # 使用明確匯入的模型
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 7)  # 假設這裡有 7 個類別
model = model.to(device)

# 計算模型的參數和 FLOPs
input_size = (3, 224, 224)  # 假設影像為 224x224
calculate_model_complexity(model, input_size, device)
