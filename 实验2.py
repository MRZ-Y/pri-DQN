import gym
from ray import tune
from ray.rllib.algorithms.ddpg import DDPGConfig
from ray.rllib.algorithms import Algorithm

# --------------------- 1. 训练配置 ---------------------
config = (
    DDPGConfig()
    .environment(env="Pendulum-v1")
    .framework("torch")
    .rollouts(num_rollout_workers=0)
    .exploration(
        exploration_config={
            "type": "OrnsteinUhlenbeckNoise",
            "random_timesteps": 5000,  # 初始随机探索步数
            "ou_base_scale": 0.3,      # 噪声强度
            "ou_theta": 0.2,           # OU过程衰减率
            "ou_sigma": 0.3,           # OU过程波动率
        }
    )
    .training(
        actor_lr=1e-4,      # Actor学习率
        critic_lr=1e-3,     # Critic学习率
        gamma=0.95,         # 折扣因子
        tau=0.005,          # 目标网络软更新系数
        train_batch_size=64 # 训练批次大小
    )
)

# --------------------- 2. 启动训练 ---------------------
tune.run(
    "DDPG",
    config=config.to_dict(),
    stop={"timesteps_total": 100000},  # 训练10万步
    checkpoint_freq=100,                # 每10次迭代保存一次
    local_dir="./ddpg_pendulum",       # 结果保存路径
)

# --------------------- 3. 测试训练好的策略 ---------------------
# 加载最新检查点（需替换为实际路径）
checkpoint_path = "./ddpg_pendulum/DDPG/DDPG_Pendulum-v1_XXXXX/checkpoint_000010"
algo = Algorithm.from_checkpoint(checkpoint_path)

# 可视化测试
env = gym.make("Pendulum-v1", render_mode="human")
obs, _ = env.reset()
total_reward = 0

for _ in range(200):
    action = algo.compute_single_action(obs)
    obs, reward, done, _, _ = env.step(action)
    total_reward += reward
    env.render()
    if done:
        break

print(f"Test episode reward: {total_reward}")
env.close()