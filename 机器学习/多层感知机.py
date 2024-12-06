import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from joblib import dump, load
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为宋体
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 加载数据
data = pd.read_csv('中证500指数.csv')

# 提取特征
X = data[['open', 'high', 'low']]  # 特征：开盘价，最高价，最低价
y = data['close']  # 目标：收盘价

# 定义输入的时间步长和预测未来的时间步长
n_input_days = 5  # 使用连续5天的数据作为输入
n_future = 1       # 预测未来1个时间步 或者多个时间步，修改这里即可。

# 创建特征和标签
Xs, ys = [], []
for i in range(len(X) - n_input_days - n_future + 1):
    Xs.append(X.iloc[i:(i + n_input_days)].values.flatten())
    ys.append(y.iloc[i + n_input_days:i + n_input_days + n_future])  # 输出为未来1天或多天的收盘价


# # 创建特征和标签
# n_input_days = 5  # 使用连续5天的数据作为输入
# n_future = 3       # 预测未来3个时间步
#
# Xs, ys = [], []
# for i in range(len(X) - n_input_days - n_future + 1):
#     Xs.append(X.iloc[i:(i + n_input_days)].values.flatten())
#     ys.append(y.iloc[i + n_input_days:i + n_input_days + n_future].values)  # 输出为未来3天的收盘价

# 转换为numpy数组进行训练
Xs = np.array(Xs)
ys = np.array(ys)

# 数据标准化
scaler = StandardScaler()
Xs_scaled = scaler.fit_transform(Xs)

# 划分数据集，这里简单地取最后20%作为测试集
split = int(0.8 * len(Xs_scaled))
X_train, X_test = Xs_scaled[:split], Xs_scaled[split:]
y_train, y_test = ys[:split], ys[split:]

# 创建多层感知机模型
mlp = MLPRegressor(hidden_layer_sizes=(100, 100), activation='relu', solver='adam', max_iter=1000)

# 训练模型
mlp.fit(X_train, y_train)


# 使用模型进行预测
y_train_pred = mlp.predict(X_train)
y_test_pred = mlp.predict(X_test)

# 将单步预测结果调整为二维数组，以保持兼容性
if n_future == 1:
    y_train_pred = y_train_pred.reshape(-1, 1)
    y_test_pred = y_test_pred.reshape(-1, 1)

# 训练模型部分已完成
# 保存模型
dump(mlp, 'mlp_model.joblib')  # MLP模型
dump(scaler, 'scaler.joblib')  # 也保存数据标准化器

# 计算训练集和测试集的R2和MAE
r2_train = r2_score(y_train, y_train_pred, multioutput='uniform_average')
r2_test = r2_score(y_test, y_test_pred, multioutput='uniform_average')
mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)


# 输出指标
print("训练集 R^2 Score:", r2_train)
print("测试集 R^2 Score:", r2_test)
print("训练集 MAE:", mae_train)
print("测试集 MAE:", mae_test)

# 预测未来的数据
last_input = X.iloc[-n_input_days:].values.flatten().reshape(1, -1)
last_input_scaled = scaler.transform(last_input)
future_predictions = mlp.predict(last_input_scaled)

# 输出预测结果
print("预测未来的股价：", future_predictions.flatten())

# 绘图展示训练集和测试集的结果
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.plot(y_train[:, 0], label='真实')
plt.plot(y_train_pred[:, 0], label='预测')
plt.title('训练集:真实vs预测')
# 添加x轴标签
plt.xlabel('时间')
# 添加y轴标签
plt.ylabel('收盘价')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(y_test[:, 0], label='真实')
plt.plot(y_test_pred[:, 0], label='预测')
plt.title('验证集:真实vs预测')
# 添加x轴标签
plt.xlabel('时间')
# 添加y轴标签
plt.ylabel('收盘价')
plt.legend()

plt.tight_layout()
plt.show()

