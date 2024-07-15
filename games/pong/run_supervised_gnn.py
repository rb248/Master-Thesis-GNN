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
from games.pong.pong_envs.pong_env import PongEnvTrain as PongEnvNew
from collections import OrderedDict
import wandb
from stable_baselines3.common.utils import get_schedule_fn

from games.model.policy import CustomHeteroGNN


# Initialize wandb
wandb.init(
    project="gnn_atari_pong",  # Replace with your project name
    sync_tensorboard=True,     # Automatically sync SB3 logs with wandb
    monitor_gym=True,          # Automatically log gym environments
    save_code=True             # Save the code used for this run
)

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


def make_pong_env(rank, seed=0, config=None):
    def _init():
        env = PongEnvNew(**config)
        env = TimeLimit(env, max_episode_steps=3000)  # Set a reasonable max_episode_steps
        monitor_path = os.path.join(log_dir, f"monitor_{rank}.csv")
        os.makedirs(log_dir, exist_ok=True)  # Create log directory if it doesn't exist
        env = Monitor(env, filename=monitor_path, allow_early_resets=True)
        env.seed(seed + rank)
        return env
    return _init

if __name__ == "__main__":
    params = OrderedDict([
        ('batch_size', 256),
        ('clip_range', 'lin_0.1'),
        ('ent_coef', 0.01),
        ('env_wrapper', []),  # No AtariWrapper for MlpPolicy
        ('frame_stack', 4),
        ('learning_rate', 'lin_2.5e-4'),
        ('n_envs', 16),
        ('n_epochs', 4),
        ('n_steps', 128),
        ('n_timesteps', 10000000.0),
        ('policy', 'MlpPolicy'),
        ('vf_coef', 0.5),
        ('normalize', False)
    ])

    log_dir = "./logs/Pong-GNN-training/"
    num_envs = params['n_envs']
    env_configs = [
        {"render_mode": None, "observation_type": "graph", "paddle_width": 5, "ball_speed": 4},
        {"render_mode": None, "observation_type": "graph", "paddle_width": 5, "ball_speed": 2},
        {"render_mode": None, "observation_type": "graph", "paddle_width": 2, "ball_speed": 4},
        {"render_mode": None, "observation_type": "graph", "paddle_width": 2, "ball_speed": 2},
        {"render_mode": None, "observation_type": "graph", "paddle_width": 2, "ball_speed": 4},
        {"render_mode": None, "observation_type": "graph", "paddle_width": 2, "ball_speed": 2},
        {"render_mode": None, "observation_type": "graph", "paddle_width": 5, "ball_speed": 2},
        {"render_mode": None, "observation_type": "graph", "paddle_width": 5, "ball_speed": 4}
    ]
    num_envs = len(env_configs)
    i=1
    envs = SubprocVecEnv([make_pong_env(i, config=env_configs[0]) for i in range(num_envs)])
    env = PongEnvNew(observation_type="graph", render_mode=None, paddle_width=5, ball_speed=2)
    env = TimeLimit(env, max_episode_steps=3000)  # Set a reasonable max_episode_steps
    env = Monitor(env, filename=log_dir, allow_early_resets=True)
    eval_callback = EvalCallback(env, best_model_save_path='./logs/pong-GNN-eval',
                                 log_path='./logs/pong-GNN-eval', eval_freq=5000,
                                 deterministic=True, render=False)
    # if params['normalize']:
    #     envs = VecNormalize(envs, norm_reward=True)

    policy_kwargs = dict(
        features_extractor_class=CustomHeteroGNN,
        features_extractor_kwargs=dict(
            features_dim=64,
            hidden_size=64,
            num_layer=2,
            obj_type_id='obj',
            arity_dict={'atom': 2},
            game='pong'
        ),
    )

    
    model = PPO("MlpPolicy", envs, verbose=1, policy_kwargs=policy_kwargs, device='cuda' if th.cuda.is_available() else 'cpu')

    callback = SaveOnBestTrainingRewardCallback(check_freq=10000, log_dir=log_dir)
    eval_callback = EvalCallback(env, best_model_save_path='./logs/pong-CNN-eval',
                                 log_path='./logs/pong-CNN-eval', eval_freq=5000,
                                 deterministic=True, render=False)
    #wandb_callback = WandbCallback(model_save_path="./models/", model_save_freq=5000, verbose=2)

    model.learn(total_timesteps=10000000, callback=[callback, eval_callback])
    model.save("ppo_custom_heterognn_pong")
