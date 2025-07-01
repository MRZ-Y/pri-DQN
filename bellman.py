import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 环境配置，定义了FrozenLake环境的地图布局
desc = ["SFFFFFFF",
        "FFFFFHFH",
        "FFFHFFFH",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHH",
        "FFFHFFFG", ]

# 创建FrozenLake环境的函数
def FrozenLake_env():
    return gym.make("FrozenLake-v1", desc=desc, is_slippery=False, render_mode=None)

# 定义神经网络模型，用于估计状态的价值
class ValueNetwork(nn.Module):
    def __init__(self, input_size):
        super(ValueNetwork, self).__init__()
        # 第一个全连接层，输入维度为input_size，输出维度为16
        self.fc1 = nn.Linear(input_size, 16)
        # ReLU激活函数
        self.relu = nn.ReLU()
        # 第二个全连接层，输入维度为16，输出维度为1
        self.fc2 = nn.Linear(16, 1)

    # 前向传播函数
    def forward(self, x):
        # 输入通过第一个全连接层
        x = self.fc1(x)
        # 通过ReLU激活函数
        x = self.relu(x)
        # 通过第二个全连接层
        x = self.fc2(x)
        return x

# 独热编码函数，将状态转换为独热编码形式
def one_hot_encode(state, state_size):
    # 创建一个全零的数组，长度为state_size
    one_hot = np.zeros(state_size)
    # 将对应状态的位置设置为1
    one_hot[state] = 1
    # 返回转换后的张量
    return torch.FloatTensor(one_hot).unsqueeze(0)

# 训练函数，用于训练价值网络
def train(env, value_network, gamma=0.9, theta=1e-8, epochs=10):
    # 使用Adam优化器
    optimizer = optim.Adam(value_network.parameters(), lr=0.001)
    # 使用均方误差损失函数
    criterion = nn.MSELoss()

    # 获取环境的状态空间大小
    state_size = env.observation_space.n
    # 训练循环
    for epoch in range(epochs):
        delta = 0
        # 遍历所有状态
        for s in range(state_size):
            # 获取当前状态的独热编码
            state_tensor = one_hot_encode(s, state_size)
            # 获取当前状态的价值估计
            v = value_network(state_tensor).item()

            action_values = []
            # 遍历所有动作
            for a in range(env.action_space.n):
                action_value = 0
                # 遍历所有可能的下一个状态及其概率
                for prob, next_state, reward, terminated in env.P[s][a]:
                    # 获取下一个状态的独热编码
                    next_state_tensor = one_hot_encode(next_state, state_size)
                    # 获取下一个状态的价值估计
                    next_value = value_network(next_state_tensor).item()
                    # 计算动作价值
                    action_value += prob * (reward + gamma * next_value)
                # 将动作价值添加到列表中
                action_values.append(action_value)

            # 获取最大动作价值作为目标价值
            target_value = np.max(action_values)
            # 将目标价值转换为张量
            target_tensor = torch.FloatTensor([target_value])

            # 清零梯度
            optimizer.zero_grad()
            # 获取当前状态的价值估计
            output = value_network(state_tensor)
            # 计算损失
            loss = criterion(output, target_tensor)
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()

            # 更新delta值
            delta = max(delta, np.abs(v - target_value))

        # 如果delta值小于阈值，认为收敛
        if delta < theta:
            print(f"Converged at epoch {epoch}")
            break

# 根据最优价值函数确定最优策略
def get_optimal_policy(env, value_network, gamma=0.9):
    # 初始化策略矩阵
    policy = np.zeros([env.observation_space.n, env.action_space.n])
    # 获取环境的状态空间大小
    state_size = env.observation_space.n
    # 遍历所有状态
    for s in range(state_size):
        # 获取当前状态的独热编码
        state_tensor = one_hot_encode(s, state_size)
        action_values = []
        # 遍历所有动作
        for a in range(env.action_space.n):
            action_value = 0
            # 遍历所有可能的下一个状态及其概率
            for prob, next_state, reward, terminated in env.P[s][a]:
                # 获取下一个状态的独热编码
                next_state_tensor = one_hot_encode(next_state, state_size)
                # 获取下一个状态的价值估计
                next_value = value_network(next_state_tensor).item()
                # 计算动作价值
                action_value += prob * (reward + gamma * next_value)
            # 将动作价值添加到列表中
            action_values.append(action_value)
        # 获取最大动作价值对应的动作
        best_action = np.argmax(action_values)
        # 更新策略矩阵
        policy[s, best_action] = 1
    return policy

# 主函数，用于运行整个流程
def main():
    # 创建环境
    env = FrozenLake_env()
    # 获取环境的状态空间大小
    input_size = env.observation_space.n
    # 创建价值网络
    value_network = ValueNetwork(input_size)
    # 训练价值网络
    train(env, value_network)
    # 获取最优策略
    policy = get_optimal_policy(env, value_network)

    # 重置环境，获取初始状态
    state, _ = env.reset()
    done = False
    # 运行环境，直到结束
    while not done:
        # 根据策略选择动作
        action = np.argmax(policy[state])
        # 执行动作，获取下一个状态、奖励和终止标志
        state, reward, done, _, _ = env.step(action)
        # 如果结束，根据奖励判断成功或失败
        if done:
            if reward == 1:
                print("成功到达目标！")
            else:
                print("掉入冰洞，失败！")
    # 关闭环境
    env.close()

# 主函数入口
if __name__ == "__main__":
    main()