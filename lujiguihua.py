import gym

# 环境配置
desc = ["SFFFFFFF",
        "FFFFFHFH",
        "FFFHFFFH",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHH",
        "FFFHFFFG", ]


def FrozenLake_env():
    return gym.make("FrozenLake-v1", desc=desc, is_slippery=False, render_mode=None)


def testFrozenLake_env():
    return gym.make("FrozenLake-v1", desc=desc, is_slippery=False, render_mode='human')


def dfs(env, state, path, visited):
    # 如果当前状态已经访问过，返回 None
    if state in visited:
        return None
    # 将当前状态标记为已访问
    visited.add(state)
    # 将当前状态添加到路径中
    path.append(state)
    # 注意这里，不再需要 step，因为我们从 P 里获取信息
    # observation, reward, terminated, truncated, info = env.step(state)

    # 定义四个可能的动作：左、下、右、上
    actions = [0, 1, 2, 3]
    for action in actions:
        # 尝试执行动作
        prob, next_state, reward, terminated = env.env.P[state][action][0]
        # 如果到达目标状态，返回路径
        if reward == 1:
            path.append(next_state)
            return path
        # 递归调用 dfs 函数
        result = dfs(env, next_state, path.copy(), visited.copy())
        if result is not None:
            return result
    # 如果没有找到路径，返回 None
    return None


def solve_with_dfs():
    # 创建环境
    env = FrozenLake_env()
    # 重置环境
    state, _ = env.reset()
    # 初始化路径和已访问集合
    path = []
    visited = set()
    # 调用 dfs 函数进行搜索
    solution = dfs(env, state, path, visited)
    if solution is not None:
        print("找到解决方案！路径如下：")
        print(solution)
    else:
        print("未找到解决方案。")
    # 关闭环境
    env.close()


if __name__ == "__main__":
    solve_with_dfs()