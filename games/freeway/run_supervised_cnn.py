import os
import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.results_plotter import ts2xy
from wandb.integration.sb3 import WandbCallback
from gymnasium.wrappers.time_limit import TimeLimit
from games.freeway.freeway_envs.freeway_env import FreewayEnv
from collections import OrderedDict
import wandb
from stable_baselines3.common.utils import get_schedule_fn

class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq, log_dir, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose > 0:
                        print(f"Saving new best model at {x[-1]} timesteps")
                        print(f"Saving new best model to {self.save_path}.zip")
                    self.model.save(self.save_path)
            wandb.log({"mean_reward": mean_reward, "timesteps": self.num_timesteps})
        return True

def make_freeway_env(rank, seed=0, config=None):
    def _init():
        env = FreewayEnv(**config)
        env = TimeLimit(env, max_episode_steps=3000)  # Set a reasonable max_episode_steps
        monitor_path = os.path.join(log_dir, f"monitor_{rank}.csv")
        os.makedirs(log_dir, exist_ok=True)  # Create log directory if it doesn't exist
        env = Monitor(env, filename=monitor_path, allow_early_resets=True)
        env.seed(seed + rank)
        return env
    return _init

if __name__ == "__main__":
    num_envs = 4  # Number of parallel environments

    params = OrderedDict([
        ('batch_size', 256),
        ('clip_range', get_schedule_fn(0.1)),  # Linear schedule for clip range
        ('ent_coef', 0.01),
        ('learning_rate', get_schedule_fn(2.5e-4)),  # Linear schedule for learning rate
        ('n_epochs', 4),
        ('n_steps', 128),
        ('vf_coef', 0.5),
    ])
    env_configs = [
        {"render_mode": None, "observation_type": "pixel", "car_speed": 4},
        {"render_mode": None, "observation_type": "pixel", "car_speed": 2},
        {"render_mode": None, "observation_type": "pixel", "car_speed": 4},
        {"render_mode": None, "observation_type": "pixel", "car_speed": 2},
    ]
    log_dir = "./logs/Freeway-CNN-training/"

    # Create a vectorized environment with SubprocVecEnv
    envs = SubprocVecEnv([make_freeway_env(i, config=env_configs[i]) for i in range(num_envs)])

    wandb.init(
        project="cnn_atari_freeway",  # Replace with your project name
        sync_tensorboard=True,           # Automatically sync SB3 logs with wandb
        monitor_gym=True,                # Automatically log gym environments
        save_code=True                   # Save the code used for this run
    )
    env = FreewayEnv(observation_type="pixel", render_mode="human", car_speed=2)
    env = Monitor(env, filename=log_dir, allow_early_resets=True)
    eval_callback = EvalCallback(env, best_model_save_path='./logs/freeway-CNN-eval',
                                 log_path='./logs/freeway-CNN-eval', eval_freq=5000,
                                 deterministic=True, render=False)

    device = "cuda" if th.cuda.is_available() else "cpu"

    model = PPO("CnnPolicy", envs, device=device, verbose=2)

    # SaveOnBestTrainingRewardCallback
    callback = SaveOnBestTrainingRewardCallback(check_freq=5000, log_dir=log_dir)
    # wandb_callback = WandbCallback(model_save_path="./models/", model_save_freq=5000, verbose=2)

    model.learn(total_timesteps=1000000, callback=[callback, eval_callback])
    model.save("ppo_freeway_pixel")
