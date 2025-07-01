import torch
import torch.nn as nn

# 增加层数和隐藏单元
input_size = 5
hidden_size = 5  # 增加隐藏单元数量
num_layers = 2    # 增加层数
lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

# 初始化多层隐藏状态和细胞状态
h_0 = torch.zeros(num_layers, 1, hidden_size)  # [num_layers, batch_size, hidden_size]
c_0 = torch.zeros(num_layers, 1, hidden_size)
x_t = torch.randn(1, 2, input_size)
# 单步更新
for t in range(2):
    output, (h_t, c_t) = lstm(x_t, (h_0, c_0))
    h_0, c_0 = h_t, c_t
    print(f"Time step {t+1}: output.shape = {output.shape}, h_t.shape = {h_t.shape}")