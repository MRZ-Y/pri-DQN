from agent import *
from env import testFrozenLake_env

def test_agent():
    # 直接加载保存的实例
    agent = torch.load('agent.pth')
    # 如果需要，将模型设置为评估模式
    agent.model.eval()
        # 创建测试环境
    test_env = testFrozenLake_env()
    # 环境配置
    state_size = test_env.observation_space.n  # 状态空间
    action_size = test_env.action_space.n  # 动作空间
    # 设置智能体的探索率为0，确保测试时使用的是学习到的策略
    agent.epsilon = 0.0
    
    # 重置环境并获取初始状态
    state = test_env.reset()
    if isinstance(state, tuple):
        state = state[0]
    # 将状态转换为one-hot编码
    state = np.eye(state_size)[state]
    done = False

    # 循环进行测试，直到环境结束
    while not done:
        # 根据当前状态选择动作
        action = agent.act(state)
        # 执行动作并获取下一个状态、奖励、是否结束等信息
        next_state, reward, done, _, _ = test_env.step(action)
        # 更新状态为下一个状态的one-hot编码
        state = np.eye(state_size)[next_state]

    # 关闭测试环境
    test_env.close()

if __name__ == "__main__":
    test_agent()
