
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


#env = PongEnvNew(render_mode='human', observation_type='pixel')
env = FreewayEnv(render_mode='human', observation_type='pixel')

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=128)
)

model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1 , device='mps')
model.learn(total_timesteps=1000000)
model.save("ppo_breakout_custom_cnn")   