import gym
from ray import tune
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms import Algorithm
import numpy as np
import torch
import torch.nn as nn
from ray.rllib.models import ModelCatalog
import torch.nn.functional as F
from ray.tune.callback import Callback
# from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

local_dir=r"C:\Users\ZY\Desktop\python_projects\Rainbow-dqn\python_projects\Rainbow-dqn\dqn_frozenlake"

# # 注册自定义模型
# ModelCatalog.register_custom_model("CustomRainbow", CustomRainbowModel)

# 自定义地图（可选）
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

# --------------------- 1. 配置 Rainbow DQN ---------------------
config = (
    DQNConfig()
    .environment(
        env="FrozenLake-v1",
        env_config={
            "desc": desc,
            "is_slippery": False,  # 关闭随机滑动
            "render_mode": None,   # 训练时不渲染
            # # 修改奖励函数（关键优化）
            # "reward_goal": 1.0,
            # "reward_hole": -0.1,  # 掉入洞的惩罚
            # "reward_step": -0.01  # 每步小惩罚
        }
    )
    .framework("torch")
    .rollouts(num_rollout_workers=0)  # 单机训练

    .training(
        # model={
        #     "fcnet_hiddens": [256, 128],  # 隐藏层结构
        #     "fcnet_activation": "relu",
        # },
        # Rainbow 核心组件
        dueling=True,                # Dueling DQN（支持）
        double_q=True,               # Double DQN（支持）
        noisy=False,                  # Noisy Networks（支持）
        n_step=3,                    # n-step 回报（支持）

        replay_buffer_config={
            "type": "MultiAgentPrioritizedReplayBuffer",  # 优先回放（支持）
            "capacity": 5000,  # 缓冲区容量
            "alpha": 0.6,            # 优先级指数
            "beta": -0.4,             # 重要性采样系数
        },
        lr=5e-4,                     # 学习率
        gamma=0.9,                  # 折扣因子
        training_intensity= 0.95,     # 每采样10步训练1次
        train_batch_size=64,         # 训练批次大小
        target_network_update_freq=100 ,  # 目标网络更新频率,训练步数


    )
    .exploration(
        exploration_config={
            "type": "EpsilonGreedy",
            "initial_epsilon": 1,
            "final_epsilon": 0.01,
            "epsilon_timesteps": 300000,
        }
        )
)

# --------------------- 2. 启动训练 ---------------------
tune.run(
    "DQN",
    config=config.to_dict(),
    stop={"episode_reward_mean": 0.8, "timesteps_total": 300000},  # 当平均奖励≥0.8时停止
    checkpoint_at_end=True,  # 仅保存最终模型（当达到stop条件时）
    local_dir=local_dir,
)

# --------------------- 3. 测试训练好的策略 ---------------------
# algo = Algorithm.from_checkpoint(local_dir)
#
# # 可视化测试
# env = gym.make('FrozenLake-v1', desc=desc, is_slippery=False, render_mode="human")
# obs, _ = env.reset()
# total_reward = 0
#
# for _ in range(100):
#     action = algo.compute_single_action(obs)
#     obs, reward, done, _, _ = env.step(action)
#     total_reward += reward
#     env.render()
#     if done:
#         print(f"Test reward: {total_reward}")
#         break
#
# env.close()