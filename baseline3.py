import gymnasium as gym
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack,VecTransposeImage,DummyVecEnv,VecMonitor,VecNormalize
from stable_baselines3.common.atari_wrappers import (
    NoopResetEnv,
    MaxAndSkipEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    ClipRewardEnv,
)
from stable_baselines3 import DQN

# 创建与make_env()完全等效的环境包装流程
vec_env = make_atari_env(
    env_id="BreakoutNoFrameskip-v4",
    n_envs=1,
    seed=123,
    wrapper_kwargs={
        "noop_max": 30,
        "frame_skip": 4,  # 设为1，因为我们要手动实现MaxAndSkipEnv
        "screen_size": 84,
        "clip_reward": True,  # 禁用自动clip，稍后手动添加
        "terminal_on_life_loss": True,
    },
    env_kwargs = {
        "repeat_action_probability": 0,  # 如果需要渲染
    },
)

# 手动添加所有包装器以匹配make_env()的逻辑
def wrap_env(env):
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    return env

# 应用自定义包装
vec_env = DummyVecEnv([lambda: wrap_env(env) for env in vec_env.envs])

# 图像处理流程
vec_env = VecTransposeImage(vec_env)  # (H,W,C) -> (C,H,W)
vec_env = VecFrameStack(vec_env, n_stack=4)  # (4,84,84)
vec_env = VecNormalize(vec_env, norm_obs=True,clip_obs=1.0)
print(f"环境观测空间: {vec_env.observation_space}")  # 应该显示 Box(0, 255, (4, 84, 84), uint8)
print(f"环境动作空间: {vec_env.action_space}")      # Discrete(4)

model = DQN(
    "CnnPolicy",
    vec_env,
    policy_kwargs={"net_arch": [512],  # 全连接层大小
                   "normalize_images": False},
    learning_rate=3e-4,
    buffer_size=20000,        # 回放缓冲区大小
    learning_starts=2000,    # 开始学习前的步数
    batch_size=32,           # 训练批次大小
    tau=1.0,                 # 目标网络更新率
    gamma=0.99,              # 折扣因子
    train_freq=4,            # 每4步训练一次
    gradient_steps=1,        # 每次训练梯度步数
    replay_buffer_class=None, # 使用普通回放缓冲区
    target_update_interval=100,  # 目标网络更新频率
    exploration_fraction=0.2,    # 探索比例（对应epsilon_timesteps=100000）
    exploration_initial_eps=1.0, # 初始探索率
    exploration_final_eps=0.01,  # 最终探索率
    verbose=1,
    seed=123
)

model.learn(
    total_timesteps=500000,
    log_interval=100,  # 每10个episode记录一次日志
)

model.save("./dqn_breakout_model1")  # 指定保存路径和文件名
