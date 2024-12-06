import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import matplotlib
import math

matplotlib.rc("font", family='Microsoft YaHei')
# 读取CSV文件
data = pd.read_csv('沪深300.csv')

# 提取特征和目标变量
features = data[['money', 'volume']].values   #用的特征是金钱和期卷，预测的是收盘价
target = data['close'].values

# 数据归一化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(features)
scaled_target = scaler.fit_transform(target.reshape(-1, 1))

# 划分训练集和测试集
train_size = int(len(data) * 0.83)
test_size = len(data) - train_size
train_features, test_features = scaled_features[:train_size], scaled_features[train_size:]
train_target, test_target = scaled_target[:train_size], scaled_target[train_size:]

# 将数据转换为LSTM的输入格式 (samples, time steps, features)
def create_lstm_input(data, look_back):
    x, y = [], []
    for i in range(len(data) - look_back):
        x.append(data[i:(i + look_back), :])
        y.append(data[i + look_back, 0])  # SOC作为目标变量的第一列
    return np.array(x), np.array(y)

look_back = 5  # 前10个时间步作为输入特征
train_features, train_target = create_lstm_input(train_features, look_back)
test_features, test_target = create_lstm_input(test_features, look_back)

# 构建LSTM模型
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 2)))  # 2是输入特征的维度 3个参数预测这里就改为3
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# 拟合模型
history =  model.fit(train_features, train_target, epochs=40, batch_size=8)
# 在测试集上进行预测
predicted_target = model.predict(test_features)
# 反归一化预测结果
predicted_target = scaler.inverse_transform(predicted_target)
test_target = scaler.inverse_transform(test_target.reshape(-1, 1))

# 计算R2和MAE损失
r2 = r2_score(test_target, predicted_target)
mae = mean_absolute_error(test_target, predicted_target)
mape = mean_absolute_percentage_error(test_target, predicted_target)
mse = mean_squared_error(test_target, predicted_target)
rmse = math.sqrt(mse)

print("验证R2 Score:", r2)
print("验证MAE:", mae)
print("验证MAPE:", mape)
print("验证MSE:", mse)
print("验证RMSE:", rmse)


# 绘制SOC预测结果和实际值的对比图
plt.figure(figsize=(8, 6))
plt.plot(range(len(test_target)), test_target, label='验证真实值')
plt.plot(range(len(test_target)), predicted_target, label='验证预测值')
plt.xlabel('时间')
plt.ylabel('ln500')
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
# 绘制训练集的训练结果
plt.plot(range(len(history.history['loss'])), history.history['loss'])
plt.xlabel('迭代次数')
plt.ylabel('损失')
plt.show()


# 绘制训练集的训练数据
train_predicted_target = model.predict(train_features)
train_predicted_target = scaler.inverse_transform(train_predicted_target.reshape(-1, 1))
train_target = scaler.inverse_transform(train_target.reshape(-1, 1))

# 计算R2和MAE损失
r2 = r2_score(train_target, train_predicted_target)
mae = mean_absolute_error(train_target, train_predicted_target)

print("训练R2 Score:", r2)
print("训练MAE:", mae)

plt.figure(figsize=(8, 6))
plt.plot(range(len(train_target)), train_target, label='训练集真实值')
plt.plot(range(len(train_target)), train_predicted_target, label='训练集预测值')
plt.xlabel('时间')
plt.ylabel('ln500')
plt.legend()
plt.show()
plt.savefig('训练集图片.jpg')