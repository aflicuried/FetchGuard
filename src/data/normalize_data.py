"""
对CTG数据进行标准化处理
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# 读取原始数据
data_path = os.path.join('data', 'processed', 'ctg_clean.csv')
df = pd.read_csv(data_path)

# 分离特征和标签
features = df.iloc[:, :-2]  # 除了最后两列（CLASS和NSP）
labels = df.iloc[:, -2:]     # 最后两列（CLASS和NSP）

# 标准化特征
scaler = StandardScaler()
features_normalized = scaler.fit_transform(features)

# 转换回DataFrame
df_normalized = pd.DataFrame(features_normalized, columns=features.columns)

# 将标签列添加回去
df_normalized = pd.concat([df_normalized, labels.reset_index(drop=True)], axis=1)

# 保存标准化后的数据
output_path = os.path.join('data', 'processed', 'ctg_normalized.csv')
df_normalized.to_csv(output_path, index=False)
print(f"标准化数据已保存至: {output_path}")
print(f"数据形状: {df_normalized.shape}")
