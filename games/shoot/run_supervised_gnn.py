
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch_geometric as pyg
from stable_baselines3.common.policies import ActorCriticPolicy
from games.freeway.freeway_envs.freeway_env import FreewayEnv
import pygame
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import wandb
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch_geometric as pyg
from stable_baselines3.common.policies import ActorCriticPolicy
from games.freeway.freeway_envs.freeway_env import FreewayEnv
import pygame

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
    This corresponds to the number of units for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
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
        # (n_batch, n_channel, height, width)

        features = self.cnn(observations)
        output = self.linear(features)
        return output

if __name__ == "__main__":

   
    num_envs = 4  # Number of parallel environments

    #Create a vectorized environment with DummyVecEnv
    def make_env(rank, seed=0):
        def _init():
            env = FreewayEnv(render_mode='rgb_array', observation_type='pixel')
            env.seed(seed + rank)
            return env
        return _init

    #env = DummyVecEnv([make_env(i) for i in range(num_envs)])
    #model = PPO.load("ppo_freeway_pixel")
    env = FreewayEnv(render_mode='human', observation_type='pixel')
    # env = DummyVecEnv([lambda: env])    
    # env = VecFrameStack(env, n_stack=4)
    # wandb.init(
    #     project="cnn_shoot",  # Replace with your project name
    #     sync_tensorboard=True,        # Automatically sync SB3 logs with wandb
    #     monitor_gym=True,             # Automatically log gym environments
    #     save_code=True                # Save the code used for this run
    #)
    
    device = "cuda" if th.cuda.is_available() else "cpu"
    model = PPO("CnnPolicy", env, verbose=2, device=device, )
    model.learn(total_timesteps=1000000)
    model.save("ppo_freeway_pixel")   