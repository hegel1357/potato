import matplotlib.pyplot as plt
import numpy as np

# 示例數據
num_epochs = 50
epochs = np.arange(num_epochs + 1)

# 假設有原始數據
validation_accuracy_1 = [0,0.6137, 0.7256, 0.7653, 0.7906, 0.7978, 0.8195, 0.8159, 0.8123, 0.8195, 0.8123,
                         0.8123, 0.8159, 0.8159, 0.8303, 0.8303, 0.8231, 0.8123, 0.8303, 0.8339, 0.8123,
                         0.8123, 0.8195, 0.8231, 0.8159, 0.8267, 0.8123, 0.8303, 0.8195, 0.8231, 0.8231,
                         0.8375, 0.8195, 0.8159, 0.8195, 0.8195, 0.8087, 0.8123, 0.8159, 0.8195, 0.8195,
                         0.8267, 0.8267, 0.8267, 0.8195, 0.8339, 0.8123, 0.8303, 0.8231, 0.8375, 0.8195]

validation_accuracy_2 = [0,0.5668, 0.7148, 0.7762, 0.7798, 0.8159, 0.8339, 0.8267, 0.8484, 0.8448, 0.8520,
                         0.8484, 0.8412, 0.8628, 0.8520, 0.8592, 0.8448, 0.8773, 0.8664, 0.8520, 0.8556,
                         0.8339, 0.8664, 0.8375, 0.8303, 0.8556, 0.8412, 0.8700, 0.8484, 0.8628, 0.8484,
                         0.8484, 0.8520, 0.8556, 0.8520, 0.8556, 0.8773, 0.8448, 0.8556, 0.8556, 0.8484,
                         0.8556, 0.8628, 0.8448, 0.8339, 0.8448, 0.8267, 0.8448, 0.8700, 0.8556, 0.8412]

# 繪圖
plt.figure(figsize=(10, 6))
plt.plot(epochs, validation_accuracy_1, color='blue', label='No', linewidth=2)
plt.plot(epochs, validation_accuracy_2, color='red', label='Yes', linewidth=2)

# 自訂繪圖
plt.title('Validation Accuracy Curves', fontsize=16, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.ylim(0, 1)  # 設定y軸範圍為0到1
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10)
plt.gca().set_facecolor('#f8f9fa')
for spine in plt.gca().spines.values():
    spine.set_edgecolor('#e9ecef')
plt.tight_layout()

# 顯示圖表
plt.show()