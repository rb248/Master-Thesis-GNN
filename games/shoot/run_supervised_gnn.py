import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from wandb.integration.sb3 import WandbCallback
from games.shoot.shoot_env import ShootingEnv
from games.model.policy import CustomHeteroGNN
import os
import numpy as np
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
            rewards = []
            for idx in range(self.training_env.num_envs):
                env_rewards = self.training_env.get_attr('get_rewards', indices=idx)[0]()
                rewards.extend(env_rewards)
            mean_reward = np.mean(rewards)
            if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward: {mean_reward:.2f}")

            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}")
                self.model.save(self.save_path)

        return True

# Initialize wandb
wandb.init(
    project="gnn_atari_freeway",  # Replace with your project name
    sync_tensorboard=True,        # Automatically sync SB3 logs with wandb
    monitor_gym=True,             # Automatically log gym environments
    save_code=True                # Save the code used for this run
)

# Wrap the environment
env = make_vec_env(lambda: ShootingEnv(observation_type='graph'), n_envs=4)

policy_kwargs = dict(
    features_extractor_class=CustomHeteroGNN,
    features_extractor_kwargs=dict(
        features_dim=64,
        hidden_size=64,
        num_layer=2,
        obj_type_id='obj',
        arity_dict={'atom': 2},
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
model.learn(total_timesteps=100000, callback=[WandbCallback(), save_best_callback])

model.save("ppo_custom_heterognn_freeway")
