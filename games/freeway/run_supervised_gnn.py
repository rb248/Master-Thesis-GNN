import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from wandb.integration.sb3 import WandbCallback
#from games.model.policy import CustomActorCriticPolicy
from games.freeway.freeway_envs.freeway_env import FreewayEnv
from games.model.policy import CustomCNN, CustomHeteroGNN
import numpy as np
import pygame
# #Initialize wandb
wandb.init(
    project="gnn_atari_freeway",  # Replace with your project name
    sync_tensorboard=True,        # Automatically sync SB3 logs with wandb
    monitor_gym=True,             # Automatically log gym environments
    save_code=True                # Save the code used for this run
)

# wandb.init(
#     project="cnn_g",  # Replace with your project name
#     sync_tensorboard=True,        # Automatically sync SB3 logs with wandb
#     monitor_gym=True,             # Automatically log gym environments
#     save_code=True                # Save the code used for this run
# )

# Wrap the environment 

import os
from stable_baselines3.common.callbacks import BaseCallback

class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -float('inf')

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            x, y = self.training_env.get_attr('reward')
            mean_reward = np.mean(y)
            if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward: {mean_reward:.2f}")

            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}")
                self.model.save(self.save_path)

        return True
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from wandb.integration.sb3 import WandbCallback
#from games.model.policy import CustomActorCriticPolicy
from games.freeway.freeway_envs.freeway_env import FreewayEnv
from games.model.policy import CustomCNN, CustomHeteroGNN
import pygame
import os

# Initialize wandb
wandb.init(
    project="gnn_atari_freeway",  # Replace with your project name
    sync_tensorboard=True,        # Automatically sync SB3 logs with wandb
    monitor_gym=True,             # Automatically log gym environments
    save_code=True                # Save the code used for this run
)

# Wrap the environment
env = FreewayEnv(render_mode='human', observation_type='graph')

policy_kwargs = dict(
    features_extractor_class=CustomHeteroGNN,
    features_extractor_kwargs=dict(
        features_dim=64,
        hidden_size=64,
        num_layer=2,
        obj_type_id='obj',
        arity_dict={'ChickenOnLane': 2, 'CarOnLane': 2, 'LaneNextToLane': 2},
        game='freeway'
    ),
)

# Create the PPO model with the custom feature extractor
model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=2)

# Set up log directory
log_dir = "./logs/"
os.makedirs(log_dir, exist_ok=True)

# Create and configure the custom callback
save_best_callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

# Train the model with WandbCallback and the custom callback
model.learn(total_timesteps=1000000, callback=[WandbCallback(), save_best_callback])
