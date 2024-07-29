# # import io
# import os
# import numpy as np
# from stable_baselines3 import PPO
# from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.callbacks import BaseCallback
# from wandb.integration.sb3 import WandbCallback
# from games.freeway.freeway_envs.freeway_env import FreewayEnv, FreewayEnvConstant
# from games.model.policy import CustomHeteroGNN
# from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.callbacks import EvalCallback
# from stable_baselines3.common.monitor import load_results
# from stable_baselines3.common.results_plotter import ts2xy
# import torch
# import wandb

# log_dir = "./logs/Freeway-GNN-training/"
# os.makedirs(log_dir, exist_ok=True)

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
#             x, y = ts2xy(load_results(self.log_dir), "timesteps")
#             if len(x) > 0:
#                 mean_reward = np.mean(y[-100:])
#                 if self.verbose > 0:
#                     print(f"Num timesteps: {self.num_timesteps}")
#                     print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")
#                 if mean_reward > self.best_mean_reward:
#                     self.best_mean_reward = mean_reward
#                     if self.verbose > 0:
#                         print(f"Saving new best model at {x[-1]} timesteps")
#                         print(f"Saving new best model to {self.save_path}.zip")
#                     self.model.save(self.save_path)
#                 #wandb.log({"mean_reward": mean_reward, "timesteps": self.num_timesteps})
#             else:
#                 device = "cpu"
#                 if self.verbose > 0:
#                     print("No data available for logging.")
#                 #wandb.log({"timesteps": self.num_timesteps})
#         return True

# def make_env(lanes, max_cars, car_speed, seed=0, rank=None):
#     def _init():
#         env = FreewayEnvConstant( render_mode='human', observation_type='graph')
#         monitor_path = os.path.join(log_dir, f"monitor_{rank}.csv")
#         os.makedirs(log_dir, exist_ok=True)  # Create log directory if it doesn't exist
#         env = Monitor(env, filename=monitor_path, allow_early_resets=True)
#         env.seed(seed + rank)
#         return env
#     return _init



# def main():
#     # Initialize wandb
#     wandb.init(
#         project="gnn_atari_freeway_random",  # Replace with your project name
#         sync_tensorboard=True,        # Automatically sync SB3 logs with wandb
#         monitor_gym=True,             # Automatically log gym environments
#         save_code=True                # Save the code used for this run
#     )

#     # Define environment configurations
#     #envs = SubprocVecEnv([make_env([50, 80, 120], 20, 2, rank=i) for i in range(2)]) 
#     envs = DummyVecEnv([make_env([50, 80, 120], 10, 2, rank=i) for i in range(1)])
#     #env = FreewayEnv(lanes=[50, 80, 120], max_cars=20, car_speed=2, render_mode='human', observation_type='graph')
#     policy_kwargs = dict(
#         features_extractor_class=CustomHeteroGNN,
#         features_extractor_kwargs=dict(
#             features_dim=64,
#             hidden_size=128,
#             num_layer=2,
#             obj_type_id='obj',
#             arity_dict={'ChickenOnLane': 2, 'CarOnLane': 2, 'LaneNextToLane': 2},
#             game='freeway'
#         ),
#     )
#     # check if mps is available
#     # if torch.backends.mps.is_available():
#     #     device = "mps"
#     # else:
#     #     device = "cpu" 
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     # Create the PPO model with the custom feature extractor
#     model = PPO('MlpPolicy', envs, policy_kwargs=policy_kwargs, verbose=2, device=device, ent_coef=0.01)
#     #model = PPO('MlpPolicy', env, policy_kwargs= policy_kwargs, verbose=2,)
#     #model.set_env(envs)
#     # Set up the evaluation environment and callback
#     # eval_env = FreewayEnv(lanes=[50, 80, 120], max_cars=20, car_speed=2, render_mode='human', observation_type='graph')
#     # eval_env = Monitor(eval_env)  # Apply Monitor wrapper here
#     eval_callback = EvalCallback(envs, best_model_save_path='./logs/freeway-GNN-eval',
#                                  log_path='./logs/freeway-GNN-eval', eval_freq=5000,
#                                  deterministic=True, render=False, n_eval_episodes=5)

#     #Create and configure the custom callback
#     save_best_callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

#     # Train the model with WandbCallback and the custom callback
#     model.learn(total_timesteps=1000000, callback=[WandbCallback(), save_best_callback])
#     #model.learn(total_timesteps=1000000)

#     model.save("ppo_custom_heterognn_freeway")

# if __name__ == "__main__":
#     main() 

# # import wandb
# # from stable_baselines3 import PPO
# # from stable_baselines3.common.env_util import make_vec_env
# # from wandb.integration.sb3 import WandbCallback
# # #from games.model.policy import CustomActorCriticPolicy
# # from games.freeway.freeway_envs.freeway_env import FreewayEnv
# # from games.model.policy import CustomCNN, CustomHeteroGNN
# # import pygame 
# # from stable_baselines3.common.vec_env import SubprocVecEnv
# # import os
# # from stable_baselines3.common.monitor import Monitor
# # # #Initialize wandb
# # wandb.init(
# #     project="gnn_atari_freeway",  # Replace with your project name
# #     sync_tensorboard=True,        # Automatically sync SB3 logs with wandb
# #     monitor_gym=True,             # Automatically log gym environments
# #     save_code=True                # Save the code used for this run
# # )

# # # wandb.init(
# # #     project="cnn_g",  # Replace with your project name
# # #     sync_tensorboard=True,        # Automatically sync SB3 logs with wandb
# # #     monitor_gym=True,             # Automatically log gym environments
# # #     save_code=True                # Save the code used for this run
# # # )
# # log_dir = "./logs/Freeway-GNN-training/"
# # # Wrap the environment 
# # def make_env(lanes, max_cars, car_speed, seed=0, rank=None):
# #     def _init():
# #         env = FreewayEnv(lanes=lanes, max_cars=max_cars, car_speed=car_speed, render_mode='human', observation_type='graph')
# #         monitor_path = os.path.join(log_dir, f"monitor_{rank}.csv")
# #         os.makedirs(log_dir, exist_ok=True)  # Create log directory if it doesn't exist
# #         env = Monitor(env, filename=monitor_path, allow_early_resets=True)
# #         env.seed(seed + rank)
# #         return env
# #     return _init 


# # #env = FreewayEnv(render_mode='human', observation_type='graph')
# # #env = make_vec_env(lambda: env, n_envs=8, vec_env_cls=SubprocVecEnv)
# # envs = SubprocVecEnv([make_env([50, 80, 120], 10, 2, rank=i) for i in range(16)])
# # # policy_kwargs = dict(
# # #     features_extractor_class=CustomCNN,
# # #     features_extractor_kwargs=dict(features_dim=128),
# # # )

# # policy_kwargs = dict(
# #     features_extractor_class=CustomHeteroGNN,
# #     features_extractor_kwargs=dict(
# #         features_dim=64,
# #         hidden_size=64,
# #         num_layer=2,
# #         obj_type_id='obj',
# #         arity_dict={'ChickenOnLane':2, 'CarOnLane':2, 'LaneNextToLane':2},
# #         game = 'freeway'
# #     ),
# # )

# # # # Create the PPO model with the custom feature extractor
# # model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=2, n_steps= 128)
# # # # Train the model with WandbCallback
# # model.learn(total_timesteps=1000000, callback=WandbCallback() )
# # # # Save the model
# # model.save("ppo_custom_heterognn")
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from wandb.integration.sb3 import WandbCallback
#from games.model.policy import CustomActorCriticPolicy
from games.freeway.freeway_envs.freeway_env import FreewayEnvConstant
from games.model.policy import CustomCNN, CustomHeteroGNN
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

env = FreewayEnvConstant(render_mode='human', observation_type='graph')
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
        arity_dict={'ChickenOnLane':2, 'CarOnLane':2, 'LaneNextToLane':2},
        game = 'freeway'
    ),
)

# # Create the PPO model with the custom feature extractor
model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=2)
# # Train the model with WandbCallback
model.learn(total_timesteps=100000, callback=WandbCallback() )
# # Save the model
model.save("ppo_custom_heterognn")
