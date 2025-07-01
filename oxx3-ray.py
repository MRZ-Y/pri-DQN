import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
from ray import tune
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.tune.callback import Callback
from stable_baselines3.common.atari_wrappers import NoopResetEnv, MaxAndSkipEnv, FireResetEnv, EpisodicLifeEnv
from ray.tune.registry import register_env
from gymnasium.wrappers import ResizeObservation, GrayScaleObservation, FrameStack

local_dir=r"C:\Users\ZY\Desktop\python_projects\Rainbow-dqn\python_projects\Rainbow-dqn\Breakout-v5"

# 定义自定义DQN模型
# class CustomDQNModel(TorchModelV2, nn.Module):
#     def __init__(self, obs_space, action_space, num_outputs, model_config, name):
#         super().__init__(obs_space, action_space, num_outputs, model_config, name)
#         nn.Module.__init__(self)
#         # 卷积层
#         self.convs = nn.Sequential(
#             nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
#             nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
#             nn.Conv2d(64, 64, 3, stride=1), nn.ReLU()
#         )
#
#         # 全连接层
#         self.fc = nn.Sequential(nn.Linear(3136, 512), nn.ReLU())
#
#         # Dueling DQN 价值和优势流
#         self.value_stream = nn.Linear(512, 1)
#         self.advantage_stream = nn.Linear(512, action_space.n)
#
#     @override(TorchModelV2)
#     def forward(self, input_dict, state=None, seq_lens=None):
#         x = input_dict["obs"].float() / 255.0
#         print(f"输入张量形状: {x.shape}")  # 应输出 [batch_size, 4, 84, 84]
#         x = self.convs(x).view(x.size(0), -1)
#         x = self.fc(x)
#
#         # Dueling DQN 计算
#         value = self.value_stream(x)
#         advantage = self.advantage_stream(x)
#         q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
#
#         return q_values ,state
#
# # 注册自定义模型
# ModelCatalog.register_custom_model("custom_dqn", CustomDQNModel)

# def make_env(config=None):
#     env = gym.make(
#         "ALE/Breakout-v5",
#         render_mode="rgb_array",
#         obs_type="grayscale",  # 使用灰度图像
#         repeat_action_probability=0,  # 禁用 Atari 随机重复动作
#     )
#     env = gym.wrappers.RecordEpisodeStatistics(env)
#     env = NoopResetEnv(env, noop_max=30)
#     env = MaxAndSkipEnv(env, skip=4)
#     env = EpisodicLifeEnv(env)
#     if "FIRE" in env.unwrapped.get_action_meanings():
#         env = FireResetEnv(env)  # 自动执行FIRE动作初始化游戏
#     env = gym.wrappers.ResizeObservation(env, (84, 84))  # 调整图像尺寸
#     env = gym.wrappers.FrameStack(env, 4)  # 堆叠4帧以捕捉运动信息（适用于强化学习）
#     env.action_space.seed(123)
#     return env

# register_env("breakout_env", make_env)

config = (
    DQNConfig()
    .environment(
        env="ALE/Breakout-v5",
        env_config={
            # 图像预处理参数
            "framestack": True,  # 默认启用4帧堆叠
            "dim": 84,  # 默认图像尺寸84x84
            "grayscale": True,  # 默认灰度化
            "frame_skip": 4,  # 默认跳4帧
            "clip_rewards": True,  # 默认裁剪奖励到[-1, 1]
        }
    )

    .framework("torch")
    .rollouts(num_rollout_workers=0)  # 单机训练
    .resources(num_gpus=0)  # 禁用 GPU
    .training(
    # model={
    #     "custom_model": "custom_dqn",
    #     "fcnet_hiddens": [], # 必须置空以禁用默认网络
    #     "custom_model_config": {
    #         "dueling": True,
    #     },
    # },

    double_q=True,  # 启用Double DQN
    n_step=3,  # 3步回报
    lr=3e-4,  # 学习率
    gamma=0.99,  # 折扣因子
    # training_intensity=0.95,  # 每采样10步训练1次
    train_batch_size=32,  # 训练批次大小
    target_network_update_freq=100,  # 每完成 100 次梯度更新后，更新目标网络
    num_steps_sampled_before_learning_starts=2000,  # 开始学习前的步数

    # 使用RLlib内置优先回放缓冲区
    replay_buffer_config={
        "type": "MultiAgentReplayBuffer",
        "capacity": 5000,
    },


)
    .exploration(
        exploration_config={
            "type": "EpsilonGreedy",
            "initial_epsilon": 1.0,
            "final_epsilon": 0.01,
            "epsilon_timesteps": 100000,
    }
    )
)


# 训练回调函数
class MetricsCallback(Callback):
    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        episode.custom_metrics["episode_length"] = episode.length
        episode.custom_metrics["reward_per_step"] = episode.total_reward / episode.length

# 启动训练
tune.run(
    "DQN",
    config=config.to_dict(),
    stop={"episode_reward_mean": 150, "timesteps_total": 3000000},
    # callbacks=[MetricsCallback()],
    # checkpoint_at_end=True,  # 仅保存最终模型（当达到stop条件时）
    # local_dir=local_dir,
)