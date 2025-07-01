import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from model import *
import torch.nn.functional as F  # 导入 F 模块
from torch.distributions import Categorical

class SegmentTree():
    def __init__(self, size):
        self.index = 0  # 当前要插入的位置索引
        self.size = size  # 线段树的最大容量
        self.full = False  # 标记线段树是否已满
        # 计算线段树的起始索引，使得所有叶子节点都在最后一层
        self.tree_start = 2**(size-1).bit_length()-1
        # 创建和树，叶子节点存储每个经验的优先级，非叶子节点存储子节点优先级的和
        self.sum_tree = np.zeros((self.tree_start + self.size,), dtype=np.float32)
        self.data = []  # 存储实际的经验数据
        self.max = 1  # 记录当前最大的优先级

    # 将优先级的更新向上传播到树的根节点，用于update多个经验
    def _propagate(self, indices):
        # 计算父节点的索引 其中第一次输入的indices代表树叶索引
        parents = (indices - 1) // 2
        # 获取唯一的父节点索引
        unique_parents = np.unique(parents)
        # 更新父节点的值，指定节点的优先级，通过计算其子节点优先级的和来更新
        # 计算子节点的索引
        children_indices = unique_parents * 2 + np.expand_dims([1, 2], axis=1)
        # 更新当前节点的值为子节点值的和
        self.sum_tree[unique_parents] = np.sum(self.sum_tree[children_indices], axis=0)
        # 如果父节点不是根节点，则继续向上传播
        if parents[0] != 0:
            self._propagate(parents)
    # 批量更新经验优先级：修改SumTree中多个叶子节点的值，并触发更新整棵树
    def update(self, indices, values):
        # 调整values的形状以匹配sum_tree的索引结果
        values = values.squeeze()
        # 设置新的优先级值
        self.sum_tree[indices] = values
        # 传播更新
        self._propagate(indices)
        # 更新最大优先级
        self.max = max(np.max(values), self.max)

    # 向上传播更新：当某个叶子节点的优先级被修改后，递归更新其所有父节点的值。用于append单个经验
    def _propagate_index(self, index):
        # 计算父节点的索引
        parent = (index - 1) // 2
        # 计算左右子节点的索引
        left, right = 2 * parent + 1, 2 * parent + 2
        # 更新父节点的值为左右子节点值的和
        self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
        # 如果父节点不是根节点，则继续向上传播
        if parent != 0:
            self._propagate_index(parent)
    # 高效地更新单个叶子节点的优先级并维护树结构的正确性
    def _update_index(self, index, value):
        # 设置新的优先级值
        self.sum_tree[index] = value
        # 传播更新
        self._propagate_index(index)
        # 更新最大优先级
        self.max = max(value, self.max)

    # 将新的经验数据和对应的优先级添加到树中，并更新索引和最大优先级
    def append(self, data, value):
        # 如果数据长度小于容量，则添加新数据
        if len(self.data) < self.size:
            self.data.append(data)
        else:
            # 否则替换旧数据
            self.data[self.index] = data
        # 更新树中的优先级
        self._update_index(self.index + self.tree_start, value)
        # 更新索引
        self.index = (self.index + 1) % self.size
        # 标记是否已满
        self.full = self.full or self.index == 0

    # 递归地在树中查找指定值对应的叶子节点索引
    def _retrieve(self, indices, values):
        # 计算子节点的索引
        children_indices = (indices * 2 + np.expand_dims([1, 2], axis=1))
        # 如果索引对应叶子节点，直接返回
        if children_indices[0, 0] >= self.sum_tree.shape[0]:
            return indices
        # 如果子节点索引对应叶子节点，处理边界情况
        elif children_indices[0, 0] >= self.tree_start:
            children_indices = np.minimum(children_indices, self.sum_tree.shape[0] - 1)
        # 获取左子节点的值
        left_children_values = self.sum_tree[children_indices[0]]
        # 判断值在左子树还是右子树
        successor_choices = np.greater(values, left_children_values).astype(np.int32)
        # 根据判断结果选择下一个节点
        successor_indices = children_indices[successor_choices, np.arange(indices.size)]
        # 如果在右子树，需要减去左子树的值
        successor_values = values - successor_choices * left_children_values
        # 递归查找
        return self._retrieve(successor_indices, successor_values)

    # 调用_retrieve，根据指定的值查找对应的优先级、数据索引和树索引
    def find(self, values):
        # 从根节点开始查找
        indices = self._retrieve(np.zeros(values.shape, dtype=np.int32), values)
        # 计算数据索引
        data_index = indices - self.tree_start
        # 返回优先级值、数据索引和树索引
        return (self.sum_tree[indices], data_index, indices)

    # 根据data_index从经验池中取出对应数据
    def get(self, data_index):
        # 创建一个与输入相同形状的结果数组
        result = np.zeros_like(data_index, dtype=object)

        # 遍历输入数组的每个维度
        for i in range(data_index.shape[0]):
            for j in range(data_index.shape[1]):
                # 获取当前索引值
                index = data_index[i, j]
                # 从data中获取对应的值
                value = self.data[index % self.size]
                # 将结果存入结果数组
                result[i, j] = value
        return result

    # 返回SumTree根节点的值，即所有优先级的和
    def total(self):
        return self.sum_tree[0]


# 移植自原项目的 ReplayMemory 类
class ReplayMemory():
    def __init__(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.capacity = args.memory_size
        self.history = args.history_length
        self.discount = args.gamma
        self.n = args.n_step
        self.priority_weight = args.priority_weight  # β修正优先采样的对梯度的影响，越大影响越大
        self.priority_exponent = args.priority_exponent  #优先级指数，用于对优先级进行调整。α=0 时等同于均匀随机采样；α=1 时采样完全基于优先级
        self.t = 0
        self.n_step_scaling = torch.tensor([self.discount ** i for i in range(self.n)], dtype=torch.float32, device=self.device)
        self.transitions = SegmentTree(args.memory_size)  # Store transitions in a wrap-around cyclic buffer within a sum tree for querying priorities
        # 定义一个空白转移，用于填充初始状态或无效状态
        self.blank_trans = [0,np.zeros((64), dtype=np.uint8), 0, 0.0, np.zeros((64), dtype=np.uint8),1]
    # 将新的经验添加到 SegmentTree 中，并将其优先级设置为当前最大优先级
    def append(self, state, action, reward, next_state, done):
        # 添加转移数据，并设置优先级为当前最大优先级
        self.transitions.append((self.t,state, action, reward, next_state, done), self.transitions.max)  # Store new transition with maximum priority
        # 如果是终止状态，重置时间步计数器
        self.t = 0 if done else self.t + 1
        # 根据给定的索引获取转移数据，并处理空白状态

    def _get_transitions(self, idxs):
        # 计算需要的转移索引
        transition_idxs = np.arange(-self.history + 1, self.n + 1) + np.expand_dims(idxs, axis=1) #有改动idxs-self.n - 1
        # 确保索引非负
        transition_idxs = np.maximum(transition_idxs, 0)
        # 获取转移数据
        transitions = self.transitions.get(transition_idxs)
        # 检查每一行是否有元素的第一个值为0
        global rows_to_delete
        rows_to_delete = []
        for i in range(transitions.shape[0]):
            # 检查当前行的所有元素
            has_zero = False
            for t in transitions[i, 1:self.history+1]:
                # 检查t是否是可索引的，并且第一个元素是否为0
                if t[0] == 0:
                    has_zero = True
                    break
            if has_zero:
                rows_to_delete.append(i)
            # 检查未来部分是否有0开头的元素
            zero_found_at = None  # 记录发现0的列索引
            for col_idx, y in enumerate(transitions[i, self.history + 1:], start=self.history + 1):
                if y[0] == 0:
                    zero_found_at = col_idx
                    break
            # 如果在未来部分发现0，从该位置开始设置空白
            if zero_found_at is not None:
                for w in range(zero_found_at, self.history+self.n):
                    transitions[i, w] = self.blank_trans
        #     # 删除指定行
        # transitions = np.delete(transitions, rows_to_delete, axis=0)
        return transitions

    # 采样经验   将优先级总和划分为多个段，从每个段中随机采样一个经验，确保采样到的经验具有非零优先级。
    def _get_samples_from_segments(self, batch_size, p_total):
        segment_length = p_total / batch_size  # Batch size number of segments, based on sum over all probabilities
        segment_starts = np.arange(batch_size) * segment_length
        valid = False
        while not valid:
            samples = np.random.uniform(0.0, segment_length, [batch_size]) + segment_starts
            probs, idxs, tree_idxs = self.transitions.find(samples)  # 优先级值、数据索引、树索引
            if np.all(probs != 0):
                valid = True
        transitions = self._get_transitions(idxs)
        # 创建未离散化的状态和 n 步后的下一个状态
        all_states = []
        all_statesnext = []
        actions = []
        all_reward = []
        all_dones = []
        for i in range(batch_size):
            row_states = []
            row_statesnext = []
            action=[]
            row_reward = []
            row_done =[]
            for j in range(self.history+self.n):
                row_states.append(transitions[i, j][1])
                row_statesnext.append(transitions[i, j][4])
                row_reward.append(transitions[i, j][3])
                row_done.append(transitions[i, j][5])
                if j == self.history-1:
                    action.append(transitions[i, j][2])
            all_states.append(row_states)
            all_statesnext.append(row_statesnext)
            actions.append(action)
            all_reward.append(row_reward)
            all_dones.append(row_done)
        states = torch.FloatTensor(all_states)
        states = states[:, :self.history, :]
        next_states = torch.FloatTensor(all_statesnext)
        next_states = next_states[:, -1:, :]
        # 获取离散化的动作
        actions = torch.LongTensor(actions).view(batch_size, -1)
        # 计算截断的 n 步折扣回报
        rewards = torch.FloatTensor(all_reward)
        rewards = rewards[:, self.history-1:-1]
        rewards = torch.matmul(rewards, self.n_step_scaling).unsqueeze(1)
        dones = torch.LongTensor(all_dones)
        dones = dones[:, -1].unsqueeze(1)
        return probs, idxs, tree_idxs, states, actions, rewards, next_states, dones
    # 计算采样概率和重要性采样权重，返回采样的经验和对应的权重
    def sample(self, batch_size):
        p_total = self.transitions.total()  # Retrieve sum of all priorities (used to create a normalised probability distribution)
        probs, idxs, tree_idxs, states, actions, rewards, next_states, dones = self._get_samples_from_segments(batch_size, p_total)  # 优先级值、数据索引、树索引
        probs = probs / p_total  # 优先级值/总优先级值
        capacity = self.capacity if self.transitions.full else self.transitions.index
        weights = (capacity * probs) ** -self.priority_weight
        weights = torch.tensor(weights / weights.max(), dtype=torch.float32, device=self.device).unsqueeze(1)  # Normalise by max importance-sampling weight from batch
        return tree_idxs, states, actions, rewards, next_states, dones, weights
    #更新优先级
    def update_priorities(self, idxs, priorities):
        priorities = np.power(priorities, self.priority_exponent)
        self.transitions.update(idxs, priorities)

class D3QNAgent:
    def __init__(self, state_size, action_size, args):
        # 初始化智能体的参数
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayMemory(args)  # 使用优先经验回放的内存
        self.gamma = args.gamma  # 折扣因子
        self.epsilon = args.epsilon_start  # 初始探索率
        self.epsilon_min = args.epsilon_min  # 最小探索率
        self.epsilon_decay = args.epsilon_decay  # 探索率衰减
        self.learning_rate = args.lr  # 学习率
        self.norm_clip = args.norm_clip #梯度裁剪
        self.n = args.n_step
        self.batch_size = args.batch_size
        self.model = D3QN(state_size, action_size)  # 当前模型
        self.target_model = D3QN(state_size, action_size)  # 目标模型
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)  # 优化器
        self.update_target_model()  # 初始化目标模型参数
        self.target_update_freq = args.target_update  # 目标网络更新频率

    def update_target_model(self):
        # 将当前模型的参数复制到目标模型
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        # 将经验存储到回放缓冲区
        self.memory.append(state, action, reward, next_state, done)

    def act(self, state):
        # 根据当前状态选择动作
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # 随机选择动作
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(1)  # 将状态转换为Tensor
        act_values = self.model(state)  # 获取动作值
        return torch.argmax(act_values).item()  # 返回具有最大动作值的动作

    def replay(self):
        # 使用经验回放进行训练
        if self.memory.transitions.index < self.batch_size * 10:
            return None  # 如果回放缓冲区中的经验不足，则不进行训练
        idxs, states, actions, rewards, next_states, dones, weights = self.memory.sample(self.batch_size)
        current_q = self.model(states).gather(1, actions)  # 获取当前Q值
        next_actions = self.model(next_states).max(1)[1].unsqueeze(1)    # 用主网络选择下一个状态的最大动作
        next_q = self.target_model(next_states).gather(1, next_actions).detach()    # 用目标网络评估这些动作的Q值
        target_q = rewards + (1 - dones) * (self.gamma ** self.n) * next_q     # 计算目标Q值
        #在计算损失函数时，将每个经验的损失乘以对应的重要性采样权重w i ，再进行平均，得到加权后的损失用于反向传播更新网络参数 。
        global rows_to_delete
        loss = nn.MSELoss(reduction='none')(current_q, target_q)  # 计算损失，不进行求和
        weighted_loss = (weights * loss).mean()
        self.optimizer.zero_grad()  # 清零梯度
        weighted_loss.backward()  # 反向传播
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.norm_clip) # 梯度裁剪
        self.optimizer.step()
        self.memory.update_priorities(idxs, loss.detach().cpu().numpy())  # 更新优先级
        return weighted_loss.item()  # 返回损失值

