import os
import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack 
# class CustomCNN(BaseFeaturesExtractor):
#     """
#     :param observation_space: (gym.Space)
#     :param features_dim: (int) Number of features extracted.
#     This corresponds to the number of units for the last layer.
#     """

#     def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
#         super().__init__(observation_space, features_dim)
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
#         features = self.cnn(observations)
#         output = self.linear(features)
#         return output
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.results_plotter import ts2xy
from wandb.integration.sb3 import WandbCallback
from gymnasium.wrappers.time_limit import TimeLimit
from games.pong.pong_envs.pong_env import PongEnvTrain as PongEnvNew
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


def make_pong_env(rank, seed=0):
def make_pong_env(rank, seed=0, config=None):
    def _init():
        env = PongEnvNew(render_mode=None, observation_type='pixel')
        env = Monitor(env)  # Wrap the environment with Monitor for training analytics
        env.seed(seed + rank)  # Seed each environment instance for reproducibility
        env = PongEnvNew(**config)
        env = TimeLimit(env, max_episode_steps=3000)  # Set a reasonable max_episode_steps
        monitor_path = os.path.join(log_dir, f"monitor_{rank}.csv")
        os.makedirs(log_dir, exist_ok=True)  # Create log directory if it doesn't exist
        env = Monitor(env, filename=monitor_path, allow_early_resets=True)
        env.seed(seed + rank)
        return env
    return _init

if __name__ == "__main__":
    num_envs = 4  # Number of parallel environments

if __name__ == "__main__":
    num_envs = 8  # Number of parallel environments

    # Create a vectorized environment with SubprocVecEnv
    #envs = SubprocVecEnv([make_pong_env(i) for i in range(num_envs)])
    #env = PongEnvNew(render_mode='human', observation_type='pixel')
    # wandb.init(
    #     project="cnn_atari_pong",  # Replace with your project name
    #     sync_tensorboard=True,           # Automatically sync SB3 logs with wandb
    #     monitor_gym=True,                # Automatically log gym environments
    #     save_code=True                   # Save the code used for this run
    # )
    # eval_callback = EvalCallback(envs, best_model_save_path='./logs/pong',
    #                           log_path='./logs/pong', eval_freq=5000,
    #                           deterministic=True, render=False)
    device = "cuda" if th.cuda.is_available() else "cpu"
    #model = PPO.load("ppo_pong_pixel")
    env = PongEnvNew(render_mode=None, observation_type='pixel')
    # best_params = {
    # 'n_steps': 8068,
    # 'gamma': 0.8026355618405157,
    # 'learning_rate': 0.0008875374998622511,
    # 'ent_coef': 0.0002385019801356279,
    # 'clip_range': 0.30518733313398033,
    # 'n_epochs': 7,
    # 'batch_size': 424  # Adjusted batch size
    # }  # Best hyperparameters found
    
    #model = PPO("CnnPolicy", envs, verbose=2, device=device, n_steps=best_params['n_steps'], gamma=best_params['gamma'], learning_rate=best_params['learning_rate'], ent_coef=best_params['ent_coef'], batch_size=best_params['batch_size'], n_epochs=best_params['n_epochs'], clip_range=best_params['clip_range'], tensorboard_log="./logs/pong")
    model = PPO("CnnPolicy", env, verbose=2)
    model.learn(total_timesteps=5000000)
    #model.learn(total_timesteps=5000000, callback=[WandbCallback(), eval_callback])
    model.save("ppo_pong_pixel") 
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
        {"render_mode": None, "observation_type": "pixel", "paddle_width": 5, "ball_speed": 4},
        {"render_mode": None, "observation_type": "pixel", "paddle_width": 5, "ball_speed": 2},
        {"render_mode": None, "observation_type": "pixel", "paddle_width": 2, "ball_speed": 4},
        {"render_mode": None, "observation_type": "pixel", "paddle_width": 2, "ball_speed": 2},
        {"render_mode": None, "observation_type": "pixel", "paddle_width": 2, "ball_speed": 4},
        {"render_mode": None, "observation_type": "pixel", "paddle_width": 2, "ball_speed": 2},
        {"render_mode": None, "observation_type": "pixel", "paddle_width": 5, "ball_speed": 2},
        {"render_mode": None, "observation_type": "pixel", "paddle_width": 5, "ball_speed": 4}
    ]
    log_dir = "./logs/Pong-CNN-training/"

    # Create a vectorized environment with SubprocVecEnv
    envs = SubprocVecEnv([make_pong_env(i, config=env_configs[0]) for i in range(num_envs)])

    wandb.init(
        project="cnn_atari_pong",  # Replace with your project name
        sync_tensorboard=True,           # Automatically sync SB3 logs with wandb
        monitor_gym=True,                # Automatically log gym environments
        save_code=True                   # Save the code used for this run
    )
    env = PongEnvNew(observation_type="pixel", render_mode=None, paddle_width=5, ball_speed=2)
    env = TimeLimit(env, max_episode_steps=3000)  # Set a reasonable max_episode_steps
    env = Monitor(env, filename=log_dir, allow_early_resets=True)
    eval_callback = EvalCallback(env, best_model_save_path='./logs/pong-CNN-eval',
                                 log_path='./logs/pong-CNN-eval', eval_freq=5000,
                                 deterministic=True, render=False)

    device = "cuda" if th.cuda.is_available() else "cpu"

    model = PPO("CnnPolicy", envs, device=device, verbose=2)

    # SaveOnBestTrainingRewardCallback
    callback = SaveOnBestTrainingRewardCallback(check_freq=5000, log_dir=log_dir)
    # wandb_callback = WandbCallback(model_save_path="./models/", model_save_freq=5000, verbose=2)

    model.learn(total_timesteps=1000000, callback=[callback, eval_callback])
    model.save("ppo_pong_pixel")
