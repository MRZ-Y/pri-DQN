from random import sample, choice, randrange, uniform
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import cv2

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
        self.convs = nn.Sequential(nn.Conv2d(1, 8, 4, stride=4, padding=0), nn.ReLU(),
                                   nn.Conv2d(8, 16, 4, stride=4, padding=0), nn.ReLU())
        self.convs1 = nn.Sequential(nn.Conv2d(21, 8, 8, stride=4, padding=0), nn.ReLU(),
                                    nn.Conv2d(8, 16, 4, stride=2, padding=0), nn.ReLU(),
                                    nn.Conv2d(16, 16, 3, stride=1, padding=0), nn.ReLU())
        self.fc1 = nn.Linear(400, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 4)
        self.relu=nn.ReLU()

    def forward(self, x):
        #卷积特征提取
        x = self.convs(x)
        x = x.reshape(x.shape[0], -1)
        # 前向传播过程
        x = self.relu(self.fc1(x))
        # x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class DQNAgent:
    def __init__(self, input_dim=64, output_dim=4, gamma=0.9, epsilon=1, epsilon_decay=0.9998, min_epsilon=0.01,
                 learning_rate=5e-4, initial_learn=32,buffer_capacity=5000, batch_size=64,net_uprate=100,n_step=3,
                 tau=0.1):
        # 创建经验回放缓冲区
        self.buffer = ReplayMemory_Per(capacity=buffer_capacity,n_step=n_step)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.net_uprate = net_uprate
        self.tau = tau
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
        if self.buffer.size() < self.batch_size:
            return
        idxs, pts, transitions = self.buffer.sample(self.batch_size)  #返回叶子号、数据值
        if np.isnan(self.buffer.tree.total()):
            print('error')   #树优先值溢出
        # minibatch = list(zip(*transitions))
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
        return loss.mean()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

def train(num_episodes):
    random.seed(123)  # Python 随机数生成器
    np.random.seed(123)  # NumPy 随机数生成器
    torch.manual_seed(123)  # PyTorch 随机数生成器
    # 创建环境
    desc = ["SFFFFFFF",
            "FFFFFHFH",
            "FFFHFFFH",
            "FFFFFHFF",
            "FFFHFFFF",
            "FHHFFFHF",
            "FHFFHFHH",
            "FFFHFFFG", ]
    env = gym.make('FrozenLake-v1', desc=desc, is_slippery=False, render_mode='rgb_array')
    state_size = env.observation_space.n     #状态空间
    action_size = env.action_space.n         #动作空间
    # 创建DQN Agent
    agent = DQNAgent()
    total_reward = 0
    for episode in range(1,num_episodes+1):
        _ = env.reset()
        frame = env.render()  # 获取当前帧
        frame = cv2.resize(frame, (128, 128), interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # 转为灰度
        frame = np.expand_dims(frame, axis=-1)  # 添加通道维度
        frame = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1) / 255.0
        done = False  # 初始化游戏结束标志位
        up=0
        while not done:
            up+=1
            action = agent.select_action(frame)
            _, reward, done, _, _ = env.step(action)
            next_frame = env.render()
            next_frame = cv2.resize(next_frame, (128, 128), interpolation=cv2.INTER_AREA)
            next_frame = cv2.cvtColor(next_frame, cv2.COLOR_RGB2GRAY)  # 转为灰度
            next_frame = np.expand_dims(next_frame, axis=-1)  # 添加通道维度
            next_frame = torch.tensor(next_frame, dtype=torch.float32).permute(2, 0, 1) / 255.0
            total_reward += reward
            if done:
                if  reward>0:
                    reward = reward + 5
                else:
                    reward = reward - 1
            else:
                reward =-0.001
            agent.store_transition((frame.numpy(), action, reward, next_frame.numpy(),done))  # 同样将动作转换为Python整数
            frame = next_frame
            if up==100:
                break
        agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.min_epsilon) #更新探索率
        loss=agent.train()
        if episode%agent.net_uprate==0:
            agent.update_target_network()  # 更新目标网络
            print(f"Episode: {episode}, pass_rate: {(total_reward/agent.net_uprate)*100}%,  loss: {loss}")
            total_reward=0
            # agent.buffer.b-= 0.01
            # print(agent.buffer.b)
    # 保存模型
    torch.save(agent, 'agent.pth')
    env.close()


def test_cv():
    def preprocess_frame(frame):
        frame = cv2.resize(frame, (128, 128), interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = np.expand_dims(frame, axis=-1)
        frame = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1) / 255.0
        return frame

    max_steps = 100
    current_step = 0

    agent = torch.load('agent.pth')
    agent.policy_net.eval()
    agent.epsilon = 0.01

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
    test_env = gym.make('FrozenLake-v1', desc=desc, is_slippery=False, render_mode='rgb_array')
    _ = test_env.reset()
    current_frame = test_env.render()

    cv2.namedWindow('Frozen Lake', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Frozen Lake', 600, 600)
    display_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR)
    cv2.imshow('Frozen Lake', display_frame)
    cv2.waitKey(500)

    current_frame = preprocess_frame(current_frame)
    done = False
    total_reward = 0

    while not done and current_step < max_steps:
        current_step += 1
        action = agent.select_action(current_frame)
        _, reward, done, _, _ = test_env.step(action)
        total_reward += reward

        next_frame = test_env.render()
        display_frame = cv2.cvtColor(next_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('Frozen Lake', display_frame)
        cv2.waitKey(500)

        next_frame = preprocess_frame(next_frame)
        current_frame = next_frame

    cv2.destroyAllWindows()
    test_env.close()

if __name__ == "__main__":
    train(10000)
    # test_cv()