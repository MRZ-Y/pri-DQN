import gymnasium as gym
import numpy as np
from agent import *
import argparse
from env import FrozenLake_env

# 添加参数解析
def parse_args():
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='DQN 训练参数')
    # 训练参数
    parser.add_argument('--episodes', type=int, default=5000, help='训练轮数')  # 修改此处
    parser.add_argument('--max_steps', type=int, default=100, help='每轮最大步数')
    # Agent参数
    parser.add_argument('--gamma', type=float, default=0.99, help='折扣因子')
    parser.add_argument('--epsilon_start', type=float, default=1.0, help='初始探索率')
    parser.add_argument('--epsilon_min', type=float, default=0.01, help='最小探索率')
    parser.add_argument('--epsilon_decay', type=float, default=0.999, help='探索率衰减')
    # parser.add_argument('--entropy_coef', type=float, default=0.01, help='熵正则化系数')
    # parser.add_argument('--kl_coef', type=float, default=0.1, help='KL散度系数')
    parser.add_argument('--lr', type=float, default=5e-4, help='学习率')
    parser.add_argument('--history_length', type=int, default=1, help='历史长度')
    parser.add_argument('--n_step', type=float, default=0, help='N步回报')
    parser.add_argument('--norm-clip', type=float, default=0.5, metavar='NORM', help='梯度裁剪的最大L2范数')
    parser.add_argument('--memory_size', type=int, default=5000, help='回放缓冲区大小')
    parser.add_argument('--batch_size', type=int, default=64, help='训练批次大小')
    parser.add_argument('--target_update', type=int, default=100, help='目标网络更新频率')
    parser.add_argument('--priority_weight', type=int, default=-0.4, help=' β修正优先采样的对梯度的影响，β=1时影响大,梯度稳定差')
    parser.add_argument('--priority_exponent', type=int, default=0.6, help='α=0 时等同于均匀随机采样；α=1 时采样完全基于优先级')
    parser.add_argument('--seed', type=int, default=123, help='随机种子')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # # 设置全局随机种子
    random.seed(args.seed)  # Python 随机数生成器
    np.random.seed(args.seed)  # NumPy 随机数生成器
    torch.manual_seed(args.seed)  # PyTorch 随机数生成器
    grows_to_delete = []
    env = FrozenLake_env()   # 创建环境
    # 初始化智能体
    state_size = env.observation_space.n     #状态空间
    action_size = env.action_space.n         #动作空间
    agent = D3QNAgent(state_size, action_size, args)  #选择智能体

    # 添加变量用于跟踪成功率和LOSS
    success_count = 0
    total_loss = 0
    # 循环进行训练
    for episode in range(args.episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        state = np.eye(state_size)[state]
        for step in range(args.max_steps):
            action = agent.act(state)
            next_state, reward, done, truncated, _ = env.step(action)
            next_state = np.eye(state_size)[next_state]

            if done and reward == 1:  # 到达目标
                reward = 10  # 奖励为20
                success_count += 1  # 如果成功，增加成功次数
            elif done and reward == 0:  # 掉入冰洞
                reward = -1  # 奖励为0
            else:  # 普通移动
                reward = -0.01  # 小的负奖励
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                break  # 结束当前episode
        if episode % agent.target_update_freq == 0: # 每隔一定次数更新目标网络
            agent.update_target_model()  # 更新目标网络
        if args.epsilon_start > args.epsilon_min:
            args.epsilon_start *= args.epsilon_decay  # 更新探索率
        loss = agent.replay()   # 进行训练
        #显示训练信息
        if loss is not None:
            total_loss += loss
        if (episode + 1) % 100 == 0:
            avg_loss = total_loss / 100  # 计算平均损失
            success_rate = success_count / 100  # 计算成功率
            print(f"Episode {episode + 1}, Average Loss: {avg_loss}, Success Rate: {success_rate}")
            success_count = 0
            total_loss = 0
    torch.save(agent, 'agent.pth')
    env.close()