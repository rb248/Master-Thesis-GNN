# # # import os
# # # import numpy as np
# # # import torch as th
# # # import torch.nn as nn
# # # from gymnasium import spaces
# # # from stable_baselines3 import PPO
# # # from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# # # from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
# # # from stable_baselines3.common.env_util import make_vec_env
# # # from stable_baselines3.common.monitor import Monitor
# # # from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
# # # from stable_baselines3.common.monitor import load_results
# # # from stable_baselines3.common.results_plotter import ts2xy
# # # from wandb.integration.sb3 import WandbCallback
# # # from gymnasium.wrappers.time_limit import TimeLimit
# # # from games.freeway.freeway_envs.freeway_env import FreewayEnv
# # # from collections import OrderedDict
# # # import wandb
# # # from stable_baselines3.common.utils import get_schedule_fn

# # # class SaveOnBestTrainingRewardCallback(BaseCallback):
# # #     def __init__(self, check_freq, log_dir, verbose=1):
# # #         super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
# # #         self.check_freq = check_freq
# # #         self.log_dir = log_dir
# # #         self.save_path = os.path.join(log_dir, "best_model")
# # #         self.best_mean_reward = -np.inf

# # #     def _init_callback(self) -> None:
# # #         if self.save_path is not None:
# # #             os.makedirs(self.save_path, exist_ok=True)

# # #     def _on_step(self) -> bool:
# # #         if self.n_calls % self.check_freq == 0:
# # #             x, y = ts2xy(load_results(self.log_dir), "timesteps")
# # #             if len(x) > 0:
# # #                 mean_reward = np.mean(y[-100:])
# # #                 if self.verbose > 0:
# # #                     print(f"Num timesteps: {self.num_timesteps}")
# # #                     print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")
# # #                 if mean_reward > self.best_mean_reward:
# # #                     self.best_mean_reward = mean_reward
# # #                     if self.verbose > 0:
# # #                         print(f"Saving new best model at {x[-1]} timesteps")
# # #                         print(f"Saving new best model to {self.save_path}.zip")
# # #                     self.model.save(self.save_path)
# # #             wandb.log({"mean_reward": mean_reward, "timesteps": self.num_timesteps})
# # #         return True

# # # def make_freeway_env(rank, seed=0, config=None):
# # #     def _init():
# # #         env = FreewayEnv(**config)
# # #         #env = TimeLimit(env, max_episode_steps=3000)  # Set a reasonable max_episode_steps
# # #         monitor_path = os.path.join(log_dir, f"monitor_{rank}.csv")
# # #         os.makedirs(log_dir, exist_ok=True)  # Create log directory if it doesn't exist
# # #         env = Monitor(env, filename=monitor_path, allow_early_resets=True)
# # #         env.seed(seed + rank)
# # #         return env
# # #     return _init

# # # if __name__ == "__main__":
# # #     num_envs = 1
# # #     params = OrderedDict([
# # #         ('batch_size', 256),
# # #         ('clip_range', get_schedule_fn(0.1)),  # Linear schedule for clip range
# # #         ('ent_coef', 0.01),
# # #         ('learning_rate', get_schedule_fn(2.5e-4)),  # Linear schedule for learning rate
# # #         ('n_epochs', 4),
# # #         ('n_steps', 128),
# # #         ('vf_coef', 0.5),
# # #     ])
# # #     env_configs = [
# # #         {'lanes': [50, 80, 120], 'max_cars': 10, 'car_speed': 2},
# # #         {'lanes': [50, 80], 'max_cars': 10, 'car_speed': 3},
# # #         {'lanes': [80, 120], 'max_cars': 10, 'car_speed': 4},
# # #         {'lanes': [50, 80], 'max_cars': 20, 'car_speed': 5},
# # #         {'lanes': [80, 120], 'max_cars': 20, 'car_speed': 6},
# # #         {'lanes': [50, 80], 'max_cars': 10, 'car_speed': 7},
# # #         {'lanes': [80, 120], 'max_cars': 10, 'car_speed': 6},
# # #         {'lanes': [50, 80, 120], 'max_cars': 20, 'car_speed': 7},
# # #     ]
# # #     log_dir = "./logs/Freeway-CNN-training/"

# # #     # Create a vectorized environment with SubprocVecEnv
# # #     envs = SubprocVecEnv([make_freeway_env(i, config=env_configs[0]) for i in range(num_envs)])

# # #     wandb.init(
# # #         project="cnn_atari_freeway",  # Replace with your project name
# # #         sync_tensorboard=True,           # Automatically sync SB3 logs with wandb
# # #         monitor_gym=True,                # Automatically log gym environments
# # #         save_code=True                   # Save the code used for this run
# # #     )
# # #     env = FreewayEnv(observation_type="pixel", render_mode="human", car_speed=2)
# # #     env = Monitor(env, filename=log_dir, allow_early_resets=True)
# # #     eval_callback = EvalCallback(envs, best_model_save_path='./logs/freeway-CNN-eval',
# # #                                  log_path='./logs/freeway-CNN-eval', eval_freq=5000,
# # #                                  deterministic=True, render=False)

# # #     device = "cuda" if th.cuda.is_available() else "cpu"

# # #     model = PPO("CnnPolicy", envs, device=device, verbose=2)

# # #     # SaveOnBestTrainingRewardCallback
# # #     callback = SaveOnBestTrainingRewardCallback(check_freq=5000, log_dir=log_dir)
# # #     # wandb_callback = WandbCallback(model_save_path="./models/", model_save_freq=5000, verbose=2)

# # #     model.learn(total_timesteps=1000000, callback=[callback, eval_callback])
# # #     model.save("ppo_freeway_pixel")






# # # import torch as th
# # # import torch.nn as nn
# # # from stable_baselines3 import PPO
# # # import torch_geometric as pyg
# # # from games.freeway.freeway_envs.freeway_env import FreewayEnv
# # # import pygame
# # # from stable_baselines3.common.utils import get_schedule_fn
# # # from stable_baselines3.common.callbacks import BaseCallback
# # # from stable_baselines3.common.monitor import Monitor
# # # import os
# # # import numpy as np
# # # import wandb
# # # from stable_baselines3.common.vec_env import SubprocVecEnv
# # # from stable_baselines3.common.results_plotter import ts2xy
# # # from stable_baselines3.common.monitor import load_results
# # # from stable_baselines3.common.callbacks import EvalCallback
# # # from gymnasium.wrappers.time_limit import TimeLimit



# # # class SaveOnBestTrainingRewardCallback(BaseCallback):
# # #     def __init__(self, check_freq, log_dir, verbose=1):
# # #         super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
# # #         self.check_freq = check_freq
# # #         self.log_dir = log_dir
# # #         self.save_path = os.path.join(log_dir, "best_model")
# # #         self.best_mean_reward = -np.inf

# # #     def _init_callback(self) -> None:
# # #         if self.save_path is not None:
# # #             os.makedirs(self.save_path, exist_ok=True)

# # #     def _on_step(self) -> bool:
# # #         if self.n_calls % self.check_freq == 0:
# # #             x, y = ts2xy(load_results(self.log_dir), "timesteps")
# # #             if len(x) > 0:
# # #                 mean_reward = np.mean(y[-100:])
# # #                 if self.verbose > 0:
# # #                     print(f"Num timesteps: {self.num_timesteps}")
# # #                     print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")
# # #                 if mean_reward > self.best_mean_reward:
# # #                     self.best_mean_reward = mean_reward
# # #                     if self.verbose > 0:
# # #                         print(f"Saving new best model at {x[-1]} timesteps")
# # #                         print(f"Saving new best model to {self.save_path}.zip")
# # #                     self.model.save(self.save_path)
# # #                 #wandb.log({"mean_reward": mean_reward, "timesteps": self.num_timesteps})
# # #             else:
# # #                 if self.verbose > 0:
# # #                     print("No data available for logging.")
# # #                 #wandb.log({"timesteps": self.num_timesteps})
# # #         return True


# # # def make_freeway_env(rank, seed=0, config=None):
# # #     def _init():
# # #         env = FreewayEnv(**config)
# # #         #env = TimeLimit(env, max_episode_steps=3000)  # Set a reasonable max_episode_steps
# # #         monitor_path = os.path.join(log_dir, f"monitor_{rank}.csv")
# # #         os.makedirs(log_dir, exist_ok=True)  # Create log directory if it doesn't exist
# # #         env = Monitor(env, filename=monitor_path, allow_early_resets=True)
# # #         env.seed(seed + rank)
# # #         return env
# # #     return _init

# # # if __name__ == "__main__":

   
# # #     num_envs = 2 # Number of parallel environments

# # #     # Create a vectorized environment with DummyVecEnv
# # #     # def make_env(rank, seed=0):
# # #     #     def _init():
# # #     #         env = FreewayEnv(render_mode='rgb_array', observation_type='pixel')
# # #     #         env.seed(seed + rank)
# # #     #         return env
# # #     #     return _init
# # #     log_dir = "./logs/Freeway-CNN-training/" 
# # #     env_configs = [
# # #         {'lanes': [50, 80, 120], 'max_cars': 10, 'car_speed': 2},
# # #         {'lanes': [50, 80], 'max_cars': 10, 'car_speed': 3},
# # #         {'lanes': [80, 120], 'max_cars': 10, 'car_speed': 1},
# # #         {'lanes': [50, 80], 'max_cars': 20, 'car_speed': 2},
# # #         {'lanes': [80, 120], 'max_cars': 20, 'car_speed': 3},
# # #         {'lanes': [50, 80], 'max_cars': 10, 'car_speed': 3},
# # #         {'lanes': [80, 120], 'max_cars': 10, 'car_speed': 2},
# # #         {'lanes': [50, 80, 120], 'max_cars': 20, 'car_speed': 1},
# # #     ] 
# # #     envs = SubprocVecEnv([make_freeway_env(i, config=env_configs[i]) for i in range(num_envs)])
# # #     #env = DummyVecEnv([make_env(i) for i in range(num_envs)])
# # #     env = FreewayEnv(observation_type="pixel", render_mode="human", car_speed=2, lanes=[50, 80, 120], max_cars=20)
# # #     env = TimeLimit(env, max_episode_steps=3000)  # Set a reasonable max_episode_steps
# # #     wandb.init(
# # #         project="cnn_atari_freeway",  # Replace with your project name
# # #         sync_tensorboard=True,           # Automatically sync SB3 logs with wandb
# # #         monitor_gym=True,                # Automatically log gym environments
# # #         save_code=True                   # Save the code used for this run
# # #     )
# # #     # env = FreewayEnv(observation_type="pixel", render_mode="human", car_speed=2)
# # #     # env = Monitor(env, filename=log_dir, allow_early_resets=True)
# # #     eval_callback = EvalCallback(env, best_model_save_path='./logs/freeway-CNN-eval',
# # #                                  log_path='./logs/freeway-CNN-eval', eval_freq=5000,
# # #                                  deterministic=True, render=False)

# # #     device = "cuda" if th.cuda.is_available() else "cpu"

# # #     #model = PPO("CnnPolicy", envs, device=device, verbose=2)

# # #     # SaveOnBestTrainingRewardCallback
# # #     callback = SaveOnBestTrainingRewardCallback(check_freq=5000, log_dir=log_dir)
# # #     # wandb_callback = WandbCallback(model_save_path="./models/", model_save_freq=5000, verbose=2)
# # #     device = "cuda" if th.cuda.is_available() else "cpu"
# # #     # model = PPO("CnnPolicy", 
# # #     #             envs, verbose=2,
# # #     #               device=device,
# # #     #                   batch_size=256,
# # #     #                     n_epochs=4, 
# # #     #                     clip_range=0.1,
# # #     #                       ent_coef=0.01, 
# # #     #                       learning_rate=get_schedule_fn(2.5e-4),
# # #     #                         vf_coef=0.5,
# # #     #                             n_steps=128
# # #     #                         ) 
# # #     model = PPO("CnnPolicy", envs, verbose=2, device=device)
# # #     #model = PPO.load("ppo_freeway_pixel")
# # #     #model.set_env(envs)
# # #     model.learn(total_timesteps=1000000, callback=[callback, eval_callback])
# # #     model.save("ppo_freeway_pixel")   







# # # import torch as th
# # # import torch.nn as nn
# # # from gymnasium import spaces
# # # from stable_baselines3 import PPO
# # # from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# # # import torch_geometric as pyg
# # # from stable_baselines3.common.policies import ActorCriticPolicy
# # # from games.freeway.freeway_envs.freeway_env import FreewayEnv
# # # import pygame
# # # from stable_baselines3.common.utils import get_schedule_fn
# # # from stable_baselines3.common.callbacks import BaseCallback
# # # from stable_baselines3.common.monitor import load_results
# # # from stable_baselines3.common.results_plotter import ts2xy
# # # from wandb.integration.sb3 import WandbCallback
# # # from gymnasium.wrappers.time_limit import TimeLimit
# # # from games.freeway.freeway_envs.freeway_env import FreewayEnv
# # # from collections import OrderedDict
# # # import wandb
# # # from stable_baselines3.common.utils import get_schedule_fn
# # # import os
# # # import numpy as np
# # # from stable_baselines3.common.monitor import Monitor
# # # from stable_baselines3.common.vec_env import SubprocVecEnv
# # # from stable_baselines3.common.callbacks import EvalCallback



# # # class SaveOnBestTrainingRewardCallback(BaseCallback):
# # #     def __init__(self, check_freq, log_dir, verbose=1):
# # #         super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
# # #         self.check_freq = check_freq
# # #         self.log_dir = log_dir
# # #         self.save_path = os.path.join(log_dir, "best_model")
# # #         self.best_mean_reward = -np.inf

# # #     def _init_callback(self) -> None:
# # #         if self.save_path is not None:
# # #             os.makedirs(self.save_path, exist_ok=True)

# # #     def _on_step(self) -> bool:
# # #         if self.n_calls % self.check_freq == 0:
# # #             x, y = ts2xy(load_results(self.log_dir), "timesteps")
# # #             if len(x) > 0:
# # #                 mean_reward = np.mean(y[-100:])
# # #                 if self.verbose > 0:
# # #                     print(f"Num timesteps: {self.num_timesteps}")
# # #                     print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")
# # #                 if mean_reward > self.best_mean_reward:
# # #                     self.best_mean_reward = mean_reward
# # #                     if self.verbose > 0:
# # #                         print(f"Saving new best model at {x[-1]} timesteps")
# # #                         print(f"Saving new best model to {self.save_path}.zip")
# # #                     self.model.save(self.save_path)
# # #                 #wandb.log({"mean_reward": mean_reward, "timesteps": self.num_timesteps})
# # #             else:
# # #                 if self.verbose > 0:
# # #                     print("No data available for logging.")
# # #                 #wandb.log({"timesteps": self.num_timesteps})
# # #         return True


# # # def make_freeway_env(rank, seed=0, config=None):
# # #     def _init():
# # #         env = FreewayEnv(**config)
# # #         #env = TimeLimit(env, max_episode_steps=3000)  # Set a reasonable max_episode_steps
# # #         monitor_path = os.path.join(log_dir, f"monitor_{rank}.csv")
# # #         os.makedirs(log_dir, exist_ok=True)  # Create log directory if it doesn't exist
# # #         env = Monitor(env, filename=monitor_path, allow_early_resets=True)
# # #         env.seed(seed + rank)
# # #         return env
# # #     return _init

# # # if __name__ == "__main__":

   
# # #     num_envs = 1  # Number of parallel environments
# # #     log_dir = "./logs/Freeway-CNN-training/"
# # #     # Create a vectorized environment with DummyVecEnv
# # #     # def make_env(rank, seed=0):
# # #     #     def _init():
# # #     #         env = FreewayEnv(render_mode='rgb_array', observation_type='pixel')
# # #     #         env.seed(seed + rank)
# # #     #         return env
# # #     #     return _init

# # #     # env = DummyVecEnv([make_env(i) for i in range(num_envs)]) 
# # #     env_configs = [
# # #         {'lanes': [50, 80, 120], 'max_cars': 10, 'car_speed': 2},
# # #         {'lanes': [50, 80], 'max_cars': 10, 'car_speed': 3},
# # #         {'lanes': [80, 120], 'max_cars': 10, 'car_speed': 1},
# # #         {'lanes': [50, 80], 'max_cars': 20, 'car_speed': 2},
# # #         {'lanes': [80, 120], 'max_cars': 20, 'car_speed': 3},
# # #         {'lanes': [50, 80], 'max_cars': 10, 'car_speed': 3},
# # #         {'lanes': [80, 120], 'max_cars': 10, 'car_speed': 2},
# # #         {'lanes': [50, 80, 120], 'max_cars': 20, 'car_speed': 1},
# # #     ]   
# # #     envs = SubprocVecEnv([make_freeway_env(i, config=env_configs[i]) for i in range(num_envs)])
# # #     env = FreewayEnv(observation_type="pixel", render_mode="human", car_speed=2, lanes=[50, 80], max_cars=10)
# # #     #env = FreewayEnv(render_mode='human', observation_type='pixel')
# # #     # env = DummyVecEnv([lambda: env])    
# # #     # env = VecFrameStack(env, n_stack=4)
    
# # #     device = "cuda" if th.cuda.is_available() else "cpu"
# # #     model = PPO("CnnPolicy", env, verbose=2, device=device)
# # #     env = FreewayEnv(observation_type="pixel", render_mode="human", car_speed=2)
# # #     # env = Monitor(env, filename=log_dir, allow_early_resets=True)
# # #     eval_callback = EvalCallback(env, best_model_save_path='./logs/freeway-CNN-eval',
# # #                                  log_path='./logs/freeway-CNN-eval', eval_freq=5000,
# # #                                  deterministic=True, render=False)

# # # #     device = "cuda" if th.cuda.is_available() else "cpu"

# # # #     #model = PPO("CnnPolicy", envs, device=device, verbose=2)

# # #     # SaveOnBestTrainingRewardCallback
# # #     callback = SaveOnBestTrainingRewardCallback(check_freq=3000, log_dir=log_dir)
# # #     model.learn(total_timesteps=1000000, callback=[callback, eval_callback])
# # #     #model.learn(total_timesteps=1000000)
# # #     model.save("ppo_freeway_pixel")   




# # import torch as th
# # import torch.nn as nn
# # from gymnasium import spaces
# # from stable_baselines3 import PPO
# # from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# # import torch_geometric as pyg
# # from stable_baselines3.common.policies import ActorCriticPolicy
# # from games.freeway.freeway_envs.freeway_env import FreewayEnv
# # import pygame
# # from stable_baselines3.common.vec_env import VecFrameStack
# # from stable_baselines3.common.vec_env import DummyVecEnv
# # from stable_baselines3.common.evaluation import evaluate_policy
# # from stable_baselines3.common.monitor import Monitor


# # import torch as th
# # import torch.nn as nn
# # from gymnasium import spaces
# # from stable_baselines3 import PPO
# # from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# # import torch_geometric as pyg
# # from stable_baselines3.common.policies import ActorCriticPolicy
# # from games.freeway.freeway_envs.freeway_env import FreewayEnv
# # import pygame
# # from stable_baselines3.common.utils import get_schedule_fn
# # from stable_baselines3.common.callbacks import BaseCallback
# # from stable_baselines3.common.monitor import load_results
# # from stable_baselines3.common.results_plotter import ts2xy
# # from wandb.integration.sb3 import WandbCallback
# # from gymnasium.wrappers.time_limit import TimeLimit
# # from stable_baselines3.common.vec_env import SubprocVecEnv
# # import os
# # import numpy as np
# # from stable_baselines3.common.callbacks import EvalCallback
# # import wandb

# # class CustomCNN(BaseFeaturesExtractor):
# #     """
# #     :param observation_space: (gym.Space)
# #     :param features_dim: (int) Number of features extracted.
# #     This corresponds to the number of units for the last layer.
# #     """

# #     def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
# #         super().__init__(observation_space, features_dim)
# #         # We assume CxHxW images (channels first)
# #         n_input_channels = observation_space.shape[0]
# #         print(f"n_input_channels: {n_input_channels}")
# #         self.cnn = nn.Sequential(
# #             nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
# #             nn.ReLU(),
# #             nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
# #             nn.ReLU(),
# #             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
# #             nn.ReLU(),
# #             nn.Flatten(),
# #         )

# #         # Compute shape by doing one forward pass
# #         with th.no_grad():
# #             sample_input = th.as_tensor(observation_space.sample()[None]).float() 

# #             n_flatten = self.cnn(sample_input).view(-1).shape[0]

# #         self.linear = nn.Sequential(
# #             nn.Linear(n_flatten, features_dim),
# #             nn.ReLU()
# #         )

# #     def forward(self, observations: th.Tensor) -> th.Tensor:
# #         # (n_batch, n_channel, height, width)

# #         features = self.cnn(observations)
# #         output = self.linear(features)
# #         return output
    

# # class SaveOnBestTrainingRewardCallback(BaseCallback):
# #     def __init__(self, check_freq, log_dir, verbose=1):
# #         super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
# #         self.check_freq = check_freq
# #         self.log_dir = log_dir
# #         self.save_path = os.path.join(log_dir, "best_model")
# #         self.best_mean_reward = -np.inf

# #     def _init_callback(self) -> None:
# #         if self.save_path is not None:
# #             os.makedirs(self.save_path, exist_ok=True)

# #     def _on_step(self) -> bool:
# #         if self.n_calls % self.check_freq == 0:
# #             x, y = ts2xy(load_results(self.log_dir), "timesteps")
# #             if len(x) > 0:
# #                 mean_reward = np.mean(y[-10:])
# #                 if self.verbose > 0:
# #                     print(f"Num timesteps: {self.num_timesteps}")
# #                     print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")
# #                 if mean_reward > self.best_mean_reward:
# #                     self.best_mean_reward = mean_reward
# #                     if self.verbose > 0:
# #                         print(f"Saving new best model at {x[-1]} timesteps")
# #                         print(f"Saving new best model to {self.save_path}.zip")
# #                     self.model.save(self.save_path)
# #                 #wandb.log({"mean_reward": mean_reward, "timesteps": self.num_timesteps})
# #             else:
# #                 if self.verbose > 0:
# #                     print("No data available for logging.")
# #                 #wandb.log({"timesteps": self.num_timesteps})
# #         return True

# # def make_freeway_env(rank, seed=0, config=None):
# #     def _init():
# #         env = FreewayEnv(**config)
# #         #env = TimeLimit(env, max_episode_steps=3000)  # Set a reasonable max_episode_steps
# #         monitor_path = os.path.join(log_dir, f"monitor_{rank}.csv")
# #         os.makedirs(log_dir, exist_ok=True)  # Create log directory if it doesn't exist
# #         env = Monitor(env, filename=monitor_path, allow_early_resets=True)
# #         env.seed(seed + rank)
# #         return env
# #     return _init


# # if __name__ == "__main__":

# #     wandb.init(
# #         project="cnn_atari_freeway_random",  # Replace with your project name
# #         sync_tensorboard=True,           # Automatically sync SB3 logs with wandb
# #         monitor_gym=True,                # Automatically log gym environments
# #         save_code=True                   # Save the code used for this run
# #     )
# #     num_envs = 8  # Number of parallel environments
# #     log_dir = "./logs/Freeway-CNN-training/"
# #     # Create a vectorized environment with DummyVecEnv
# #     # def make_env(rank, seed=0):
# #     #     def _init():
# #     #         env = FreewayEnv(render_mode='rgb_array', observation_type='pixel')
# #     #         env.seed(seed + rank)
# #     #         return env
# #     #     return _init

# #     # env = DummyVecEnv([make_env(i) for i in range(num_envs)])
# #     #env = FreewayEnv(render_mode='human', observation_type='pixel') 
# #     env_configs = [
# #         {'lanes': [50, 80, 120], 'max_cars': 10, 'car_speed': 2},
# #         {'lanes': [50, 80], 'max_cars': 10, 'car_speed': 1},
# #         {'lanes': [80, 120], 'max_cars': 10, 'car_speed': 1},
# #         {'lanes': [50, 80], 'max_cars': 15, 'car_speed': 2},
# #         {'lanes': [80, 120], 'max_cars': 15, 'car_speed': 1},
# #         {'lanes': [50, 80], 'max_cars': 10, 'car_speed': 1},
# #         {'lanes': [80, 120], 'max_cars': 10, 'car_speed': 2},
# #         {'lanes': [50, 80, 120], 'max_cars': 15, 'car_speed': 1},
# #     ]   
# #     envs = SubprocVecEnv([make_freeway_env(i, config=env_configs[i]) for i in range(num_envs)])
# #     # env = DummyVecEnv([lambda: env])    
# #     # env = VecFrameStack(env, n_stack=4)
# #     eval_callback = EvalCallback(envs, best_model_save_path='./logs/freeway-CNN-eval',
# #                                  log_path='./logs/freeway-CNN-eval', eval_freq=3000,
# #                                  deterministic=True, render=False)

# # #     device = "cuda" if th.cuda.is_available() else "cpu"

# # #     #model = PPO("CnnPolicy", envs, device=device, verbose=2)

# #     # SaveOnBestTrainingRewardCallback
# #     callback = SaveOnBestTrainingRewardCallback(check_freq=3000, log_dir=log_dir)
# #     device = "cuda" if th.cuda.is_available() else "cpu"
# #     model = PPO("CnnPolicy", envs, verbose=2, device=device, n_steps=128)
# #     model.learn(total_timesteps=1000000, callback=[callback, eval_callback])
# #     model.save("ppo_freeway_pixel")   


# import os
# import torch as th
# from stable_baselines3 import PPO
# from stable_baselines3.common.callbacks import EvalCallback, CallbackList
# from stable_baselines3.common.vec_env import SubprocVecEnv
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.results_plotter import ts2xy
# import wandb
# import numpy as np
# from stable_baselines3.common.evaluation import evaluate_policy
# from stable_baselines3.common.callbacks import BaseCallback
# from stable_baselines3.common.monitor import load_results
# from games.freeway.freeway_envs.freeway_env import FreewayEnv
# import csv
# import pandas as pd

# # Curriculum phases
# curriculum_phases = [
#     {'lanes': [50,80, 120], 'max_cars': 10, 'car_speed': 1},  # Easiest configuration
#     {'lanes': [50, 80], 'max_cars': 10, 'car_speed': 2}, 
#     {'lanes': [50, 80, 120], 'max_cars': 10, 'car_speed': 3},
#     {'lanes': [50, 80, 120], 'max_cars': 15, 'car_speed': 2}, 
#     {'lanes': [50, 80, 120], 'max_cars': 15, 'car_speed': 3},  # More complex configuration
# ]

# class SaveOnBestTrainingRewardCallback(BaseCallback):
#     def __init__(self, check_freq, log_dir, verbose=1):
#         super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
#         self.check_freq = check_freq
#         self.log_dir = log_dir
#         self.save_path = os.path.join(log_dir, "best_model")
#         self.best_mean_reward = -np.inf

#     def _init_callback(self) -> None:
#         if self.save_path is not None:
#             os.makedirs(self.save_path, exist_ok=True)

#     def _on_step(self) -> bool:
#         if self.n_calls % self.check_freq == 0:
#             try:
#                 x, y = ts2xy(load_results(self.log_dir), "timesteps")
#                 if len(x) > 0:
#                     mean_reward = np.mean(y[-10:])
#                     if self.verbose > 0:
#                         print(f"Num timesteps: {self.num_timesteps}")
#                         print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")
#                     if mean_reward > self.best_mean_reward:
#                         self.best_mean_reward = mean_reward
#                         if self.verbose > 0:
#                             print(f"Saving new best model at {x[-1]} timesteps")
#                             print(f"Saving new best model to {self.save_path}.zip")
#                         self.model.save(self.save_path)
#                     with open(os.path.join(self.log_dir, 'monitor.csv'), 'a') as f:
#                         writer = csv.writer(f)
#                         writer.writerow([self.num_timesteps, mean_reward, x[-1]])
#                 else:
#                     if self.verbose > 0:
#                         print("No data available for logging.")
#             except pd.errors.ParserError as e:
#                 print(f"Error reading the CSV file: {e}")
#                 with open(os.path.join(self.log_dir, 'monitor.csv'), 'r') as f:
#                     lines = f.readlines()
#                     for i, line in enumerate(lines):
#                         if len(line.split(',')) != 3:
#                             print(f"Malformed line at index {i}: {line}")
#                 return True
#         return True

# def make_freeway_env(rank, seed=0, config=None):
#     def _init():
#         env = FreewayEnv(**config)
#         monitor_path = os.path.join(log_dir, f"monitor_{rank}.csv")
#         os.makedirs(log_dir, exist_ok=True)
#         env = Monitor(env, filename=monitor_path, allow_early_resets=True)
#         env.seed(seed + rank)
#         return env
#     return _init

# if __name__ == "__main__":
#     wandb.init(
#         project="cnn_atari_freeway_random",
#         sync_tensorboard=True,
#         monitor_gym=True,
#         save_code=True
#     )
#     num_envs = 1  # A reasonable number of parallel environments
#     log_dir = "./logs/Freeway-CNN-training/"
    
#     # Initial training with the easiest configuration
#     initial_env_config = curriculum_phases[0]
#     envs = SubprocVecEnv([make_freeway_env(i, config=initial_env_config) for i in range(num_envs)]) 
#     #envs = FreewayEnv(**initial_env_config)
#     device = "cuda" if th.cuda.is_available() else "cpu"
#     callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
#     eval_callback = EvalCallback(envs, best_model_save_path='./logs/freeway-CNN-eval',
#                                  log_path='./logs/freeway-CNN-eval', eval_freq=1000,
#                                  deterministic=True, render=False)
#     callbacks = CallbackList([callback, eval_callback])
    
#     model = PPO("CnnPolicy", envs, verbose=2, device=device, n_steps=128)
#     model.learn(total_timesteps = 100000)
#     model.save("ppo_freeway_phase_0")
    
#     # Curriculum learning loop
#     for phase_index, phase in enumerate(curriculum_phases[1:], start=1):
#         print(f"Training with configuration: {phase}")
        
#         envs = SubprocVecEnv([make_freeway_env(i, config=phase) for i in range(num_envs)])
#         model.set_env(envs)
        
#         model.learn(total_timesteps=50000)
#         model.save(f"ppo_freeway_phase_{phase_index}")

#         # Optional: Evaluate the agent on all previous phases to ensure skill retention
#         for previous_phase_index in range(phase_index + 1):
#             previous_phase = curriculum_phases[previous_phase_index]
#             eval_env = make_freeway_env(0, config=previous_phase)()
#             mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
#             print(f"Evaluation on phase {previous_phase_index}: Mean reward: {mean_reward} ± {std_reward}")


 
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
import pygame
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import os
from stable_baselines3.common.vec_env import SubprocVecEnv

# pul

# if __name__ == "__main__":

#     log_dir = "./logs/Freeway-CNN-training/"
#     num_envs = 1  # Number of parallel environments

#     # Create a vectorized environment with DummyVecEnv
#     # def make_env(rank, seed=0):
#     #     def _init():
#     #         env = FreewayEnv(render_mode='rgb_array', observation_type='pixel')
#     #         env.seed(seed + rank)
#     #         return env
#     #     return _init

#     # env = DummyVecEnv([make_env(i) for i in range(num_envs)])
#     env = FreewayEnv(render_mode='human', observation_type='pixel')
#     env = SubprocVecEnv([make_freeway_env(i) for i in range(num_envs)])
#     # env = DummyVecEnv([lambda: env])    
#     # env = VecFrameStack(env, n_stack=4)
    
#     device = "cuda" if th.cuda.is_available() else "cpu"
#     model = PPO("CnnPolicy", env, verbose=2, device=device)
#     model.learn(total_timesteps=1000000)
#     model.save("ppo_freeway_pixel") 


import random
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
import torch
from skimage.color import rgb2gray
from skimage.transform import resize
from collections import deque
import os
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.results_plotter import ts2xy 
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.a2c import A2C


# Your FreewayEnv class definition goes here
class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq, log_dir, reward_threshold, verbose=1):
            super().__init__(verbose)
            self.check_freq = check_freq
            self.log_dir = log_dir
            self.save_path = os.path.join(log_dir, "best_model")
            self.best_mean_reward = -np.inf
            self.reward_threshold = reward_threshold

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print("Num timesteps: {}".format(self.num_timesteps))
                    print(
                        "Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(
                            self.best_mean_reward, mean_reward
                        )
                    )

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print("Saving new best model at {} timesteps".format(x[-1]))
                        print("Saving new best model to {}.zip".format(self.save_path))
                    self.model.save(self.save_path)
                if mean_reward > self.reward_threshold:
                    print(f"Stopping training because the mean reward {mean_reward:.2f} exceeded the threshold {self.reward_threshold:.2f}")
                    return False

        return True

class StopTrainingOnRewardThreshold(BaseCallback):
    def __init__(self, reward_threshold, log_dir, check_freq=1000, verbose=1):
        super(StopTrainingOnRewardThreshold, self).__init__(verbose)
        self.reward_threshold = reward_threshold
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.best_mean_reward = -np.inf
        self.episode_rewards = deque(maxlen=100)
        self.save_path = os.path.join(log_dir, "best_model")

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                print(y[:-100])
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
                if mean_reward > self.reward_threshold:
                    print(f"Stopping training because the mean reward {mean_reward:.2f} exceeded the threshold {self.reward_threshold:.2f}")
                    return False
        return True

def make_freeway_env(rank, config, seed=0):
    def _init():
        env = FreewayEnv(**config)
        monitor_path = os.path.join(log_dir, f"monitor_{rank}.csv")
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, filename=monitor_path, allow_early_resets=True)
        env.seed(seed + rank)
        return env
    return _init

if __name__ == "__main__":
    total_timesteps_per_stage = 500000
    reward_threshold = 100
    log_dir = "./logs/Freeway-CNN-training/"

    

    curriculums = [
        {'lanes': [50, 80, 120], 'max_cars': 10, 'car_speed': 1},
        {'lanes': [50, 80], 'max_cars': 15, 'car_speed': 2},
        {'lanes': [50, 120], 'max_cars': 20, 'car_speed': 2},
        {'lanes': random.choice([[50, 80], [50, 120], [80, 120]]), 'max_cars': random.choice([10, 15, 20]), 'car_speed': random.choice([1, 2, 3])}
    ]

    env = DummyVecEnv([make_freeway_env(i, curriculums[0]) for i in range(1)])
    #model = PPO("CnnPolicy", env, verbose=1, ent_coef=0.01, learning_rate=get_schedule_fn(2.5e-4) , n_steps=128)
    model = PPO("CnnPolicy", env, verbose=1, ent_coef = 0.01)
    for i, curriculum in enumerate(curriculums):
        print(f"Training with curriculum {i+1}...")
        stop_callback = SaveOnBestTrainingRewardCallback(reward_threshold=reward_threshold, verbose=1, log_dir=log_dir, check_freq=1000)
        model.learn(total_timesteps=total_timesteps_per_stage, callback=stop_callback)

        model.save(f"ppo_freeway_curriculum_{i+1}")

        if i < len(curriculums) - 1:
            env.close()
            env = SubprocVecEnv([make_freeway_env(j, curriculums[i+1]) for j in range(8)])
            model.set_env(env)

    print("Training complete.")
    env.close()
