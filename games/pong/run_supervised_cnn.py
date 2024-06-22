import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
from games.pong.pong_envs.pong_env import PongEnvNew
from stable_baselines3.common.env_util import make_vec_env

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
    This corresponds to the number of units for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        print(f"n_input_channels: {n_input_channels}")
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            sample_input = th.as_tensor(observation_space.sample()[None]).float() 
            n_flatten = self.cnn(sample_input).view(-1).shape[0]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        features = self.cnn(observations)
        output = self.linear(features)
        return output

def make_env(rank, seed=0):
    def _init():
        env = PongEnvNew(render_mode='rgb_array', observation_type='pixel')
        env.seed(seed + rank)
        return env
    return _init

if __name__ == '__main__':
    num_envs = 4  # Number of parallel environments

    # Create a vectorized environment with DummyVecEnv
    env = DummyVecEnv([make_env(i) for i in range(num_envs)])

    model = PPO("CnnPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)

    # Save the model
    model.save("ppo_custom_env")

    # Load the model
    model = PPO.load("ppo_custom_env")

    # Create a single instance of the environment for evaluation
    eval_env = PongEnvNew(render_mode='human', observation_type='pixel')

    # Evaluate the policy
    from stable_baselines3.common.evaluation import evaluate_policy
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)

    print(f"Mean reward: {mean_reward} ± {std_reward}")
