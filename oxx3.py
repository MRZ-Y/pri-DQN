import time
from gymnasium.wrappers import FrameStack, ResizeObservation, GrayScaleObservation
from stable_baselines3.common.atari_wrappers import NoopResetEnv, MaxAndSkipEnv, FireResetEnv,EpisodicLifeEnv
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import cv2
from random import uniform
import os

class SumTree:
    def __init__(self, capacity):
        # 创建一个完全二叉树,树节点数是capacity的2倍-1
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.write = 0


    # update to the root node
    def _propagate(self, idx, change):
        # 初始化树的数组representation和存储经验数据的数组。更新idx叶节点的值,并递归更新父节点
        parent = (idx - 1) // 2  #idx为叶子号，parent为父亲结点号
        self.tree[parent] += change #更新父亲值（p）
        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        # 根据值s递归查找叶节点（号）索引
        left = 2 * idx + 1  # 已知父亲结点idx可以找到两个子结点
        right = left + 1
        if left >= len(self.tree):
            return idx              #返回叶子号
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        #返回整棵树的前缀和,也就是所有样本优先级值之和。
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        # 插入新样本的优先级p和数据data到树中
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        # 更新索引idx样本的优先级
        change = p - self.tree[idx]
        self.tree[idx] = p  #更新叶子值（p）
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        # 根据随机数s,返回样本节点号，节点值（优先级），数据值
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], dataIdx)
    def getdata(self , dataIdx):
        return self.data[dataIdx]

class ReplayMemory_Per(object):
    # stored as ( s, a, r, s_ ) in SumTree
    # 初始化树和相关参数,保存经验池容量,以及优先级相关的参数a和e。
    def __init__(self, n_step,capacity=10000, a=0.6, e=0.01,b=-0.4 ):
        self.tree = SumTree(capacity)
        self.memory_size = capacity
        self.prio_max = 1
        self.a = a
        self.e = e
        self.b = b
        self.n_step = n_step

    def push(self, data):
        # 插入新样本,优先级为最大值
        p = (np.abs(self.prio_max) + self.e) ** self.a  # 比例优先权
        self.tree.add(p, data)

    def sample(self, batch_size):
        # 按处于[0, 总优先级]区间的随机数,采样样本索引
        idxs = []
        pt=[]
        segment = self.tree.total() / batch_size
        sample_datas = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = uniform(a, b)
            idx, p, dataIdx  = self.tree.get(s)
            # 构建一个N步经验
            sequence_datas = []
            for step in range(self.n_step):
                if (dataIdx + step)<self.tree.n_entries:
                    # 获取每一步的经验
                    sequence_datas.append(self.tree.getdata(dataIdx + step))
                else:
                    break
            sample_datas.append(sequence_datas)
            pt.append(p)
            idxs.append(idx)

        pt=pt / self.tree.total()
        pts=(self.tree.n_entries*pt)**self.b
        return idxs,pts, sample_datas   #返回batch_size个叶子号，数据值

    def update(self, idxs, errors):
        # 更新索引样本的优先级
        errors= np.abs(errors)
        self.prio_max = max(self.prio_max, max(errors))
        for i, idx in enumerate(idxs):
            p = (errors[i] + self.e) ** self.a
            self.tree.update(idx, p)

    def size(self):
        # 返回样本数量
        return self.tree.n_entries

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        # self.convs = nn.Sequential(nn.Conv2d(1, 8, 4, stride=4, padding=0), nn.ReLU(),
        #                            nn.Conv2d(8, 16, 4, stride=4, padding=0), nn.ReLU())
        self.convs1 = nn.Sequential(nn.Conv2d(4, 32, 8, stride=4, padding=0), nn.ReLU(),
                                    nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
                                    nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU())
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 4)
        self.relu=nn.ReLU()

    def forward(self, x):
        #卷积特征提取
        x = self.convs1(x)
        x = x.reshape(x.shape[0], -1)
        # 前向传播过程
        x = self.relu(self.fc1(x))
        # x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class DQNAgent:
    def __init__(self, input_dim=64, output_dim=4, gamma=0.99, epsilon=1, epsilon_tough=1, duration=30000, min_epsilon=0.01,
                 learning_rate=1e-4, initial_learn=3000,buffer_capacity=30000, batch_size=32,net_uprate=100,n_step=1,
                 tau=0.1):
        # 创建经验回放缓冲区
        self.buffer = ReplayMemory_Per(capacity=buffer_capacity,n_step=n_step)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_tough = epsilon_tough
        self.duration = duration
        self.min_epsilon = min_epsilon
        self.slope = (self.min_epsilon - self.epsilon_tough) / self.duration  # 计算衰减斜率（负数，表示下降）
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.net_uprate = net_uprate
        self.tau = tau
        self.initial_learn=initial_learn
        # 创建DQN网络和目标网络
        self.policy_net = DQN().to(self.device)
        self.target_net = DQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        # 定义优化器和损失函数
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def store_transition(self, data):
        self.buffer.push(data)

    def select_action(self, state):
        if np.random.uniform() < self.epsilon:
            return np.random.randint(self.output_dim)
        else:
            with torch.no_grad():
                return self.policy_net(state.unsqueeze(0)).data.max(1)[1].item()

    def train(self):
        if self.buffer.size() < self.initial_learn:
            return 0
        idxs, pts, transitions = self.buffer.sample(self.batch_size)  #返回叶子号、数据值
        if np.isnan(self.buffer.tree.total()):
            print('error')   #树优先值溢出
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        for batch in transitions:
            # 提取states和actions，只取step(0)的状态
            states.append(batch[0][0])
            actions.append(batch[0][1])
            # 初始化奖励、下一个状态和完成标志
            reward_sum = batch[0][2]
            current_next_state = batch[0][3]
            current_done = batch[0][4]
            for i, (_, _, reward, next_state, done) in enumerate(batch):
                if done:
                    break
                # 更新奖励
                if i != 0:
                    reward_sum += reward * (self.gamma ** i)
                # 更新下一个状态和完成标志
                current_next_state = next_state
                current_done = done

            rewards.append(reward_sum)
            next_states.append(current_next_state)
            dones.append(current_done)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.LongTensor(np.array(dones)).unsqueeze(1)

        state_action_values = self.policy_net(states).gather(1, actions)

        next_action_batch = torch.unsqueeze(self.policy_net(next_states).max(1)[1], 1)
        next_state_values = self.target_net(next_states).detach().gather(1, next_action_batch)
        expected_state_action_values = (1-dones) * (next_state_values * self.gamma) + rewards

        td_errors = (state_action_values - expected_state_action_values).detach().squeeze().tolist()
        self.buffer.update(idxs, td_errors)  # update td error和优先级
        loss = F.mse_loss(state_action_values, expected_state_action_values, reduction='none')
        pts = torch.tensor(pts / pts.max(), dtype=torch.float32).unsqueeze(1)
        weighted_loss = (pts  * loss).mean()
        self.optimizer.zero_grad()
        weighted_loss.backward()
        self.optimizer.step()
        return loss.detach().mean()

    def linear_schedule(self ,t):
        return max(self.slope * t + 1, self.min_epsilon)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# 定义环境包装
def make_env(env_id, seed=123, render_mode="human", repeat_prob=0):
    # 基础环境创建
    env = gym.make(env_id, render_mode=render_mode, obs_type="grayscale",repeat_action_probability=repeat_prob)
    # 记录episode统计信息（奖励、长度等）
    env = gym.wrappers.RecordEpisodeStatistics(env)
    # 标准Atari环境包装流程
    env = NoopResetEnv(env, noop_max=30)  # 随机执行0-30次noop动作
    env = MaxAndSkipEnv(env, skip=4)  # 每4帧取最大值并执行一次动作
    env = EpisodicLifeEnv(env)  # 生命结束时视为 episode 结束
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)  # 自动执行FIRE动作初始化游戏
    env = gym.wrappers.ResizeObservation(env, (84, 84))  # 调整图像尺寸
    env = gym.wrappers.FrameStack(env, 4)  # 堆叠4帧以捕捉运动信息（适用于强化学习）
    # 固定环境动作空间的随机种子
    env.action_space.seed(seed)
    return env

def train(num_episodes):
    # 创建Breakout-v4环境，设置为RGB模式
    env = make_env("ALE/Breakout-v5", render_mode='rgb_array')
    random.seed(123)  # Python 随机数生成器
    np.random.seed(123)  # NumPy 随机数生成器
    torch.manual_seed(123)  # PyTorch 随机数生成器
    agent = DQNAgent()
    # 游戏总奖励
    total_reward = 0
    mean_timestep = []
    mean_loss = []
    for episode in range(1,num_episodes+1):
        # prev_lives = 5
        # 重置环境，获取初始观察
        obs, _ = env.reset()
        # obs = np.expand_dims(obs, axis=0) #扩展维度
        obs = torch.tensor(np.array(obs), dtype=torch.float32) / 255.0
        # save_reward = 0
        # 最大时间步
        max_timesteps = 3000
        # 时间步计数器
        timestep = 0
        while timestep < max_timesteps:
            if timestep % 4 == 0:
                loss = agent.train()
                mean_loss.append(loss)
            action = agent.select_action(obs)
            # 执行动作，获取新状态、奖励和终止信息
            next_obs, reward, done, _, info = env.step(action)
            # next_obs = np.expand_dims(next_obs, axis=0) #扩展维度
            # 转换为BGR（OpenCV默认格式）
            # img_bgr = np.transpose(next_obs, (1, 2, 0))
            # # 保存图像到桌面
            # cv2.imwrite(f"C:/Users/ZY/Desktop/breakout_frame_{timestep}.jpg", img_bgr)
            # print(f"已保存图像: {timestep}")
            next_obs =  torch.tensor(np.array(next_obs), dtype=torch.float32) / 255.0
            # print("状态空间形状:", obs.shape)
            # 累加奖励
            total_reward += reward
            # save_reward += reward
            # if info['lives'] < prev_lives:    # 检测死亡：生命减少
            #     prev_lives = info['lives']
            #     save_reward = - 1
                # if all(np.array_equal(x,state_queue[0]) for x in state_queue):
                #     save_reward = save_reward - 4
            agent.store_transition((obs.numpy(), action,reward, next_obs.numpy(),done))
            obs = next_obs
            # 检查游戏是否结束
            timestep += 1
            if done:
                mean_timestep.append(timestep)
                break
        agent.epsilon = agent.linear_schedule(episode)  # 更新探索率
        # loss = agent.train()
        if episode % agent.net_uprate == 0:
            agent.update_target_network()  # 更新目标网络
            agent.buffer.b = max(agent.buffer.b - 0.001,-1)
            print(f"Episode: {episode},  reward: {total_reward/100},  mean_timestep: {np.array(mean_timestep).mean()},  loss: {np.array(mean_loss).mean():.4f},  b: {agent.buffer.b:.4f}")
            mean_timestep = []
            mean_loss = []
            total_reward = 0
    # 保存模型
    torch.save(agent.policy_net.state_dict(), os.path.join("Breakout-v5_dqn_model-pri.pth"))
    # 关闭环境
    env.close()
    print(f"游戏结束，总奖励: {total_reward}")
    return

# def trainagain(num_episodes):
#     # 创建Breakout-v4环境，设置为RGB模式
#     env = make_env("ALE/Breakout-v5", render_mode='rgb_array')
#     random.seed(123)  # Python 随机数生成器
#     np.random.seed(123)  # NumPy 随机数生成器
#     torch.manual_seed(123)  # PyTorch 随机数生成器
#     # 创建agent并加载预训练模型
#     agent = DQNAgent(input_dim=64, output_dim=4, gamma=0.99, epsilon=1, epsilon_tough=1, duration=12000, min_epsilon=0.01,
#                  learning_rate=3e-4, initial_learn=3000,buffer_capacity=20000, batch_size=32,net_uprate=100,n_step=1,
#                  tau=0.1)
#     agent.policy_net.load_state_dict(torch.load('Breakout-v5_dqn_model-pri.pth'))
#     agent.target_net.load_state_dict(agent.policy_net.state_dict())  # 同时更新目标网络
#     # 游戏总奖励
#     total_reward = 0
#     mean_timestep = []
#     mean_loss = []
#     gxjs = 0
#     for episode in range(1, num_episodes + 1):
#         obs, _ = env.reset()
#         obs = torch.tensor(np.array(obs), dtype=torch.float32) / 255.0
#         # 最大时间步
#         max_timesteps = 3000
#         timestep = 0
#         while timestep < max_timesteps:
#             if timestep % 4 == 0:
#                 gxjs += 1
#                 loss = agent.train()
#                 mean_loss.append(loss)
#                 if gxjs % agent.net_uprate == 0:
#                     agent.update_target_network()  # 更新目标网络
#                     gxjs = 0
#             action = agent.select_action(obs)
#             next_obs, reward, done, _, info = env.step(action)
#             next_obs = torch.tensor(np.array(next_obs), dtype=torch.float32) / 255.0
#             total_reward += reward
#             agent.store_transition((obs.numpy(), action, reward, next_obs.numpy(), done))
#             obs = next_obs
#             timestep += 1
#             if done:
#                 mean_timestep.append(timestep)
#                 break
#         # agent.epsilon = agent.linear_schedule(episode)  # 更新探索率
#         if  episode % 100 == 0:
#             agent.buffer.b = max(agent.buffer.b - 0.001, -1)
#             print(
#                 f"Episode: {episode},  reward: {total_reward / 100},  mean_timestep: {np.array(mean_timestep).mean()},  loss: {np.array(mean_loss).mean():.4f},  b: {agent.buffer.b:.2f}")
#             mean_timestep = []
#             mean_loss = []
#             total_reward = 0
#
#     torch.save(agent.policy_net.state_dict(), os.path.join("Breakout-v5_dqn_model-pri_continued.pth"))
#     env.close()
#     return

def test_cv():
    # 创建Breakout-v4环境，设置为RGB模式
    test_env = make_env("ALE/Breakout-v5")
    random.seed(123)  # Python 随机数生成器
    np.random.seed(123)  # NumPy 随机数生成器
    torch.manual_seed(123)  # PyTorch 随机数生成器
    agent = DQNAgent(epsilon=0,min_epsilon=0)
    agent.policy_net.load_state_dict(torch.load('Breakout-v5_dqn_model-pri-5.pth'))
    agent.policy_net.eval()
    obs, _ = test_env.reset()
    obs = torch.tensor(np.array(obs), dtype=torch.float32) / 255.0
    max_timesteps = 10000
    timestep = 0
    prev_lives = 5  # 初始生命数
    star = False
    while timestep < max_timesteps:
        action = agent.select_action(obs)
        if star:
            action = 1
            star = False
        next_obs, reward, done, _, info = test_env.step(action)
        current_lives = info['lives']
        if current_lives < prev_lives:
            prev_lives = current_lives
            print(f"生命损失！当前生命: {prev_lives}")
            star = True
        next_obs = torch.tensor(np.array(next_obs), dtype=torch.float32) / 255.0
        obs = next_obs
        # # 控制帧率，使游戏可视化更清晰
        time.sleep(0.05)
        timestep += 1
        if reward>0:
            print(f"当前奖励: {reward}")
        if done:
            break
    test_env.close()
    return

if __name__ == "__main__":
    # train(50000)
    test_cv()