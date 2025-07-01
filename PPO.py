import gym
from gym import spaces
import numpy as np
import ray
from ray import tune
from ray.rllib.env.env_context import EnvContext
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.utils.framework import try_import_tf
import os
from datetime import datetime

tf1, tf, tfv = try_import_tf()


# 自定义环境包装器
class CustomCartPoleWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.observation_space = spaces.Box(
            low=np.array([-4.8, -np.inf, -0.418, -np.inf, 0, 0]),
            high=np.array([4.8, np.inf, 0.418, np.inf, 100, 100]),
            dtype=np.float32
        )

    def reset(self):
        obs = self.env.reset()
        extended_obs = np.append(obs, [0, 0])
        return extended_obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        original_obs_len = self.env.observation_space.shape[0]

        custom_features = obs[original_obs_len:] if len(obs) > original_obs_len else [0, 0]
        step_count = custom_features[0] if len(custom_features) > 0 else 0
        cumulative_reward = custom_features[1] if len(custom_features) > 1 else 0

        step_count += 1
        cumulative_reward += reward

        extended_obs = np.append(obs[:original_obs_len], [step_count, cumulative_reward])
        modified_reward = reward + 0.1 * (1 if abs(obs[2]) < 0.1 else -1)

        return extended_obs, modified_reward, done, info


# 自定义模型
class CustomModel(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        self.original_obs_space = gym.spaces.Box(
            low=obs_space.low[:4],
            high=obs_space.high[:4],
            dtype=np.float32
        )

        self.custom_obs_space = gym.spaces.Box(
            low=obs_space.low[4:],
            high=obs_space.high[4:],
            dtype=np.float32
        )

        self.original_net = FullyConnectedNetwork(
            self.original_obs_space, action_space, num_outputs, model_config, name + "_original"
        )

        self.custom_net = FullyConnectedNetwork(
            self.custom_obs_space, action_space, num_outputs, model_config, name + "_custom"
        )

        input_size = num_outputs * 2
        self.output_layer = tf.keras.layers.Dense(
            num_outputs,
            activation=None,
            kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.01)

    def forward(self, input_dict, state, seq_lens):
        original_obs = input_dict["obs"][:, :4]
        custom_obs = input_dict["obs"][:, 4:]

        original_out, _ = self.original_net({"obs": original_obs}, state, seq_lens)
        custom_out, _ = self.custom_net({"obs": custom_obs}, state, seq_lens)

        combined = tf.concat([original_out, custom_out], axis=1)
        output = self.output_layer(combined)

        return output, []


# 注册自定义模型
ModelCatalog.register_custom_model("custom_model", CustomModel)


# 修改后的环境创建函数
def env_creator(env_config: EnvContext):
    env = gym.make("CartPole-v1")
    return CustomCartPoleWrapper(env)


# 配置和训练
if __name__ == "__main__":
    ray.init()

    # 创建合法的实验名称
    experiment_name = f"PPO_CartPole_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    config = {
        "env": "CartPole-v1",  # 使用字符串而不是函数引用
        "env_config": {},  # 添加空的环境配置
        "model": {
            "custom_model": "custom_model",
            "fcnet_hiddens": [64, 64],
            "fcnet_activation": "relu",
        },
        "framework": "torch",
        "num_workers": 1,
        "num_gpus": 0,
        "lr": 1e-3,
        "gamma": 0.99,
        "train_batch_size": 200,
        "rollout_fragment_length": 50,
    }

    # 确保结果目录存在
    os.makedirs("./ray_results", exist_ok=True)

    # 使用PPO算法
    analysis = tune.run(
        "PPO",
        name=experiment_name,  # 指定合法的实验名称
        config=config,
        stop={"episode_reward_mean": 195},
        checkpoint_at_end=True,
        local_dir="./ray_results",
        verbose=1,
    )

    ray.shutdown()