import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 定義混淆矩陣
conf_matrix = np.array([
    [56,  0,  0,  0,  0,  0,  1],
    [ 0, 71,  0,  0,  4,  0,  0],
    [ 0,  0, 20,  0,  0,  0,  1],
    [ 1,  0,  0,  5,  1,  0,  0],
    [ 0,  4,  2,  0, 55,  0,  1],
    [ 0,  5,  0,  0,  0, 30,  0],
    [ 1,  1,  5,  1,  1,  0, 45]
])

# 定義類別標籤
class_names = ['Bacteria', 'Fungi', 'Healthy', 'Nematode', 'Pest', 'Phytopthora', 'Virus']

# 創建熱圖，調整字體大小
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names,
            annot_kws={"size": 16})  # 調整數字大小

plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# 調整布局以確保標籤完全顯示
plt.tight_layout()

# 顯示圖形
plt.show()