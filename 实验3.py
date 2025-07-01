import gym
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

# 定义一个添加噪声的网络层
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.6):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def update_std_init(self, new_std_init):
        """更新噪声标准差"""
        self.std_init = new_std_init
        self.reset_parameters()

    def forward(self, input):
        if self.training:
            return F.linear(input,
                           self.weight_mu + self.weight_sigma * self.weight_epsilon,
                           self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)

class NoisyD3QN(nn.Module):
    """带NoisyNet的Dueling Double DQN网络"""
    def __init__(self, state_size, action_size):
        super(NoisyD3QN, self).__init__()
        self.action_size = action_size

        # 噪声网络结构 (全部使用NoisyLinear)
        self.fc1 = NoisyLinear(64, 512)          # 共享特征层
        self.fc2 = NoisyLinear(512, 256)          # 共享特征层

        self.value = NoisyLinear(256, 1)          # 输出状态价值
        self.adv = NoisyLinear(256, action_size)  # 输出动作优势

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        value = self.value(x)  # 标量状态价值
        adv = self.adv(x)        # 每个动作的优势

        # 合并双流：Q = V + A - mean(A)
        adv_mean = torch.mean(adv, dim=1, keepdim=True)
        q_values = value + adv - adv_mean
        return q_values

    def reset_noise(self):
        """重置所有Noisy层的噪声"""
        for module in [self.fc1, self.fc2, self.value, self.adv]:
            module.reset_noise()


class NoisyD3QNAgent:
    """NoisyNet+DuelingDDQN智能体"""
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)  # 经验回放缓冲区
        self.gamma = 0.99    # 折扣因子
        self.batch_size = 64  # 批大小
        # 添加衰减参数
        self.initial_std_init = 0.6
        self.final_std_init = 0.01
        self.std_decay_episodes = 8000  # 线性衰减到最终值的总轮数
        self.current_episode = 0  # 当前轮次
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 主网络和目标网络 (均含Noisy层)
        self.model = NoisyD3QN(state_size, action_size).to(self.device)
        self.target_model = NoisyD3QN(state_size, action_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.update_target_model()  # 初始同步权重

    def update_target_model(self):
        """硬更新目标网络"""
        self.target_model.load_state_dict(self.model.state_dict())

    def update_noise_std(self):
        """线性衰减噪声标准差"""
        self.current_episode += 1
        # 计算衰减比例（0到1之间）
        decay_ratio = min(1.0, self.current_episode / self.std_decay_episodes)
        # 线性插值：initial → final
        self.current_std = self.initial_std_init - decay_ratio * (self.initial_std_init - self.final_std_init)

        # 更新所有噪声层
        for model in [self.model, self.target_model]:
            for module in [model.fc1, model.fc2, model.value, model.adv]:
                module.update_std_init(self.current_std)

    def act(self, state):
        # 将numpy数组形式的状态转换为PyTorch张量，并添加批次维度
        state = torch.FloatTensor(state).unsqueeze(0)
        # 前向传播获取Q值
        q_value = self.model(state)
        # 选择Q值最大的动作索引（dim=1表示在动作维度上取最大值）
        action = q_value.max(1)[1].data[0]         # .data[0]用于获取tensor中的值
        # 将动作索引从GPU（如果有）移回CPU，并转换为numpy数组
        action = action.cpu().numpy()
        # 将numpy数组转换为整数类型，以便输入到gym环境中
        action = int(action)
        # 返回选择的动作
        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """用经验回放训练网络"""
        if len(self.memory) < self.batch_size:
            return

        # 从内存中采样
        minibatch = random.sample(self.memory, self.batch_size)
        states = torch.tensor([t[0] for t in minibatch], dtype=torch.float).to(self.device)
        actions = torch.tensor([t[1] for t in minibatch], dtype=torch.long).to(self.device)
        rewards = torch.tensor([t[2] for t in minibatch], dtype=torch.float).to(self.device)
        next_states = torch.tensor([t[3] for t in minibatch], dtype=torch.float).to(self.device)
        dones = torch.tensor([t[4] for t in minibatch], dtype=torch.long).to(self.device)

        # 计算当前Q值 (使用主网络)
        current_q = self.model(states).gather(1, actions.unsqueeze(1))

        # 计算目标Q值 (使用目标网络 + Double DQN技巧)
        next_actions = self.model(next_states).argmax(1)  # 主网络选择动作
        next_q = self.target_model(next_states).gather(1, next_actions.unsqueeze(1))  # 目标网络评估
        target_q = rewards.unsqueeze(1) + self.gamma * next_q * (1 - dones.unsqueeze(1))

        # 计算MSE损失
        loss = F.mse_loss(current_q, target_q.detach())

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 重置所有Noisy层的噪声 (关键步骤!)
        self.model.reset_noise()
        self.target_model.reset_noise()

def train_agent(env, agent, episodes=10000):
    total_reward = 0
    for e in range(episodes):
        agent.update_noise_std()
        if e % 500 == 0:  # 每500轮打印一次噪声强度
            print(f"Episode {e}, Noise STD: {agent.current_std:.4f}")
        state = env.reset()
        state = state[0] if isinstance(state, tuple) else state
        state = np.eye(state_size)[state]
        done = False
        up=0

        while not done:
            up+=1
            action = agent.act(state)
            next_state, reward, done, truncated, info = env.step(action)
            next_state = np.eye(state_size)[next_state]
            total_reward += reward
            if done:
                if reward > 0:
                    reward = reward + 5
                else:
                    reward = reward - 1
            else:
                reward = -0.001
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if up == 100:
                break
            if done:
                break

        agent.replay()

        if e % 100 == 0:
            agent.update_target_model()  # 更新目标网络
            print(f"Episode: {e}, pass_rate: {(total_reward/100)*100}%")
            total_reward=0
    return

desc = [
    "SFFFFFFF",
    "FFFFFHFH",
    "FFFHFFFH",
    "FFFFFHFF",
    "FFFHFFFF",
    "FHHFFFHF",
    "FHFFHFHH",
    "FFFHFFFG",
]

# Create environment
env = gym.make('FrozenLake-v1', desc=desc,is_slippery=False)
state_size = env.observation_space.n
action_size = env.action_space.n

# Initialize agent
agent = NoisyD3QNAgent(state_size, action_size)

# Train the agent
train_agent(env, agent, episodes=10000)