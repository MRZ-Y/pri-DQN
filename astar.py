import heapq
import gym

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

# 启发式函数，使用曼哈顿距离计算从当前状态到目标状态的距离
def heuristic(state, goal_state, env):
    # 获取环境的行数和列数
    rows, cols = env.env.nrow, env.env.ncol
    # 计算当前状态的行和列
    current_row, current_col = state // cols, state % cols
    # 计算目标状态的行和列
    goal_row, goal_col = goal_state // cols, goal_state % cols
    # 返回曼哈顿距离
    return abs(current_row - goal_row) + abs(current_col - goal_col)

# A*搜索算法的实现
def a_star_search(env):
    # 重置环境，获取初始状态
    start_state, _ = env.reset()
    # 找到目标状态
    for state in env.env.P:
        for action in env.env.P[state]:
            for prob, next_state, reward, terminated in env.env.P[state][action]:
                if reward == 1:
                    goal_state = next_state
                    break

    # 初始化开放列表（优先队列），存储待探索的状态及其相关信息
    open_list = []
    heapq.heappush(open_list, (0, start_state, []))
    # 初始化关闭集合，存储已探索的状态
    closed_set = set()

    # 开始搜索
    while open_list:
        # 从开放列表中取出f值最小的状态
        _, current_state, path = heapq.heappop(open_list)
        # 如果当前状态已经在关闭集合中，跳过
        if current_state in closed_set:
            continue
        # 将当前状态加入关闭集合
        closed_set.add(current_state)
        # 更新路径
        path = path + [current_state]

        # 如果当前状态是目标状态，返回路径
        if current_state == goal_state:
            return path

        # 遍历所有可能的动作（上下左右）
        for action in range(4):
            # 获取下一个状态、奖励和终止标志
            prob, next_state, reward, terminated = env.env.P[current_state][action][0]
            # 如果下一个状态不在关闭集合中
            if next_state not in closed_set:
                # 计算g值（当前路径长度）
                g = len(path)
                # 计算h值（启发式函数值）
                h = heuristic(next_state, goal_state, env)
                # 计算f值（g + h）
                f = g + h
                # 将下一个状态加入开放列表
                heapq.heappush(open_list, (f, next_state, path))

    # 如果未找到解决方案，返回None
    return None

# 使用A*算法解决问题的函数
def solve_with_a_star():
    # 创建环境
    env = FrozenLake_env()
    # 调用A*搜索
    solution = a_star_search(env)
    if solution is not None:
        print("找到解决方案！路径如下：")
        print(solution)
    else:
        print("未找到解决方案。")
    # 关闭环境
    env.close()

# 主函数入口
if __name__ == "__main__":
    solve_with_a_star()