import torch
import torch.nn as nn
import torch.nn.functional as F  # 导入 F 模块

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(64, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 4)
        self.relu=nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.relu(self.fc2(x))
        x = self.fc4(x)
        return x

class D3QN(nn.Module):
    def __init__(self, state_size, action_size):
        super(D3QN, self).__init__()
        # 定义三层全连接层
        self.fc1 = nn.Linear(state_size, 512)
        self.fc3 = nn.Linear(512, 128)
        # 定义价值流和优势流
        self.value_stream = nn.Linear(128, 1)
        self.advantage_stream = nn.Linear(128, action_size)

    def forward(self, state):
        # 处理短序列：如果序列长度小于最小长度，进行填充
        # if state.size(1) < 4:
        #     padding_length = 4 - state.size(1)
        #     state = F.pad(state, (0, 0, padding_length, 0), "constant", 0)
        # state=state.view(state.shape[0], -1)
        # 前向传播过程
        x = torch.relu(self.fc1(state))  # 第一层全连接层
        x = torch.relu(self.fc3(x))  # 第三层全连接层
        value = self.value_stream(x)  # 价值流
        advantages = self.advantage_stream(x)  # 优势流
        qvals = value + (advantages - advantages.mean(dim=1, keepdim=True))  # 计算Q值
        qvals=qvals.squeeze()
        return qvals

class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        # 状态值函数分支
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        # 优势函数分支
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # 计算状态值函数和优势函数
        value = self.value_stream(x)
        advantages = self.advantage_stream(x)
        # 组合成 Q 值
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values

class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc3 = nn.Linear(256, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        action_probs = F.softmax(self.fc3(x), dim=-1)
        return action_probs

class ValueNetwork(nn.Module):
    def __init__(self, state_size):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        value = self.fc2(x)
        return value