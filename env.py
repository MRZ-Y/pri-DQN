import gymnasium as gym
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