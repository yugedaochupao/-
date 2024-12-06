import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 加载数据
data = pd.read_csv('沪深300.csv')

# 计算相关系数矩阵
correlation_matrix = data[['close', 'open', 'high', 'low', 'volume', 'money']].corr()

# 绘制热力图
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('relative map')
plt.show()