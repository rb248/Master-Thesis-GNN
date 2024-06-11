import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from wandb.integration.sb3 import WandbCallback
#from games.model.policy import CustomActorCriticPolicy
from games.pong.pong_envs.pong_env import PongEnvNew
from games.model.policy import CustomCNN, CustomHeteroGNN
# Initialize wandb
# wandb.init(
#     project="gnn_atari",  # Replace with your project name
#     sync_tensorboard=True,        # Automatically sync SB3 logs with wandb
#     monitor_gym=True,             # Automatically log gym environments
#     save_code=True                # Save the code used for this run
# )

# wandb.init(
#     project="cnn_g",  # Replace with your project name
#     sync_tensorboard=True,        # Automatically sync SB3 logs with wandb
#     monitor_gym=True,             # Automatically log gym environments
#     save_code=True                # Save the code used for this run
# )

# Wrap the environment 

env = PongEnvNew(render_mode='human', observation_type='graph')
# policy_kwargs = dict(
#     features_extractor_class=CustomCNN,
#     features_extractor_kwargs=dict(features_dim=128),
# )

policy_kwargs = dict(
    features_extractor_class=CustomHeteroGNN,
    features_extractor_kwargs=dict(
        features_dim=64,
        hidden_size=64,
        num_layer=2,
        obj_type_id='obj',
        arity_dict={'atom': 2}
    ),
)

# Create the PPO model with the custom feature extractor
model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=1)
# Train the model with WandbCallback
#model.learn(total_timesteps=1000000, callback=WandbCallback())
model.learn(total_timesteps=1000000)
# Save the model
#model.save("ppo_custom_heterognn")
#model.save("ppo_custom_cnn")
# wandb.save("ppo_custom_heterognn.zip")  # Save the model to wandb

# # Load the model
# #model = PPO.load("ppo_custom_heterognn")
# model = PPO.load("ppo_custom_cnn")
# # Evaluate the model
# obs = env.reset()
# for _ in range(1000):
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()

# import torch as th
# import torch.nn as nn
# from gymnasium import spaces

# from stable_baselines3 import PPO
# from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# import torch_geometric as pyg
# from stable_baselines3.common.policies import ActorCriticPolicy
# from games.pong.pong_envs.pong_env import PongEnvNew

# class CustomCNN(BaseFeaturesExtractor):
#     """
#     :param observation_space: (gym.Space)
#     :param features_dim: (int) Number of features extracted.
#     This corresponds to the number of units for the last layer.
#     """

#     def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
#         super().__init__(observation_space, features_dim)
#         # We assume CxHxW images (channels first)
#         n_input_channels = observation_space.shape[0]
#         print(f"n_input_channels: {n_input_channels}")
#         self.cnn = nn.Sequential(
#             nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
#             nn.ReLU(),
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
#             nn.ReLU(),
#             nn.Flatten(),
#         )

#         # Compute shape by doing one forward pass
#         with th.no_grad():
#             sample_input = th.as_tensor(observation_space.sample()[None]).float() 

#             n_flatten = self.cnn(sample_input).view(-1).shape[0]

#         self.linear = nn.Sequential(
#             nn.Linear(n_flatten, features_dim),
#             nn.ReLU()
#         )

#     def forward(self, observations: th.Tensor) -> th.Tensor:
#         # (n_batch, n_channel, height, width)
#         print(f"Observations shape: {observations.shape}")

#         features = self.cnn(observations)
#         output = self.linear(features)
#         return output


# env = PongEnvNew(render_mode='human', observation_type='pixel')

# policy_kwargs = dict(
#     features_extractor_class=CustomCNN,
#     features_extractor_kwargs=dict(features_dim=128)
# )

# model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
# #model = PPO("CnnPolicy", "BreakoutNoFrameskip-v4", policy_kwargs=policy_kwargs, verbose=1, batch_size=1)
# model = 
# model.learn(total_timesteps=1000000) 


