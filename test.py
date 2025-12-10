import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from models.timeshap_spring_RNN import build_model, model_features

# データ読み込み
data = pd.read_csv('csv/combined_15m_2024_spring.csv')

# スケーリング
values = data[['combined']].values
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(values)

# 特徴量とラベルの作成
sequence_length = 24
X, y = [], []
for i in range(len(data_scaled) - sequence_length):
    X.append(data_scaled[i:i+sequence_length])
    y.append(data_scaled[i+sequence_length])
X = np.array(X)
y = np.array(y)

# モデル読み込み
model = build_model(len(model_features))
model.load_state_dict(torch.load('RNN_model.pth'))
model.eval()  # 評価モードにする場合

# 26-8601 i=0:26
o = 1100
i = o - 26

X_single = X[i].reshape(1, X.shape[1], X.shape[2])  # (1, seq_len, features)

with torch.no_grad():
    X_tensor = torch.FloatTensor(X_single)
    y_pred_single, _ = model(X_tensor)  # modelのforwardは (output, hn) を返す想定
    y_pred_single = y_pred_single.numpy()

y_pred_inv = scaler.inverse_transform(y_pred_single)
y_true_inv = scaler.inverse_transform(y[i].reshape(-1, 1))

csv_index_1 = i + sequence_length + 2  # → y[i] に対応する行
csv_index_2 = i + sequence_length      # → y[i] に対応する行

abs_error = abs(y_pred_inv[0][0] - y_true_inv[0][0])


print(f"Target CSV index for prediction (starts from row 26 when i=0): {csv_index_1}")
print(f"Predicted value by the model: {y_pred_inv[0][0]:.3f}")
print(f"Actual value (ground truth): {y_true_inv[0][0]:.3f}")
print(f"Absolute error between prediction and actual: {abs_error:.3f}")
