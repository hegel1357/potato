import matplotlib.pyplot as plt
import numpy as np

# 示例數據
num_epochs = 50
epochs = np.arange(num_epochs + 1)

# 假設有原始數據
validation_accuracy_1 = [2,1.5723 ,0.8227, 0.4102, 0.1950, 0.0966, 0.0508, 0.0281, 0.0222, 0.0204,0.0203,
                         0.0213, 0.0187, 0.0173, 0.0160, 0.0161, 0.0166, 0.0152, 0.0181, 0.0165, 0.0152,
                         0.0143, 0.0162, 0.0190, 0.0146, 0.0155, 0.0150, 0.0152, 0.0150, 0.0141, 0.0172,
                         0.0148, 0.0161, 0.0156, 0.0160, 0.0143, 0.0154, 0.0142, 0.0150, 0.0159, 0.0167,
                         0.0167, 0.0152, 0.0171, 0.0149, 0.0159, 0.0165, 0.0153, 0.0154, 0.0164, 0.0168]

validation_accuracy_2 = [2,1.5940, 0.9117, 0.5947, 0.4369, 0.3538, 0.2930, 0.2462, 0.2048, 0.1976, 0.2105,
                         0.1797, 0.1794, 0.1823, 0.1731, 0.1674, 0.1707, 0.1641, 0.1594, 0.1690, 0.1608, 
                         0.1513, 0.1527, 0.1575, 0.1540, 0.1631, 0.1756, 0.1713, 0.1548, 0.1625, 0.1591, 
                         0.1590, 0.1582, 0.1566, 0.1458, 0.1534, 0.1706, 0.1572, 0.1569, 0.1557, 0.1682, 
                         0.1647, 0.1538, 0.1646, 0.1640, 0.1591, 0.1587, 0.1604, 0.1651, 0.1484, 0.1608]

# 繪圖
plt.figure(figsize=(10, 6))
plt.plot(epochs, validation_accuracy_1, color='blue', label='No', linewidth=2)
plt.plot(epochs, validation_accuracy_2, color='red', label='Yes', linewidth=2)

# 自訂繪圖
plt.title('Training Loss Curves', fontsize=16, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.ylim(0, 2)  # 設定y軸範圍為0到1
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10)
plt.gca().set_facecolor('#f8f9fa')
for spine in plt.gca().spines.values():
    spine.set_edgecolor('#e9ecef')
plt.tight_layout()

# 顯示圖表
plt.show()