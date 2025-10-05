"""
生成CTG数据的相关性热力图
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 读取数据
data_path = os.path.join('data', 'processed', 'ctg_clean.csv')
df = pd.read_csv(data_path)

# 排除最后两列 CLASS 和 NSP
df_features = df.iloc[:, :-2]

# 计算相关性矩阵
correlation_matrix = df_features.corr()

# 创建热力图
plt.figure(figsize=(16, 14))
sns.heatmap(correlation_matrix,
            annot=True,  # 显示数值
            fmt='.2f',   # 保留两位小数
            cmap='coolwarm',  # 颜色方案
            center=0,    # 以0为中心
            square=True,  # 方形单元格
            linewidths=0.5,
            cbar_kws={'shrink': 0.8})

plt.title('Correlation Matrix of CTG Features (Before Processing)',
          fontsize=16,
          pad=20)
plt.tight_layout()

# 保存图片
output_dir = os.path.join('reports', 'new')
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'correlation_matrix_heatmap.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"相关性热力图已保存至: {output_path}")

plt.close()