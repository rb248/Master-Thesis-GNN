import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
from games.pong.pong_envs.pong_env import PongEnvNew
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

def make_pong_env(rank, seed=0):
    def _init():
        env = PongEnvNew(render_mode=None, observation_type='pixel')
        env = Monitor(env)  # Wrap the environment with Monitor for training analytics
        env.seed(seed + rank)  # Seed each environment instance for reproducibility
        return env
    return _init

if __name__ == "__main__":
    num_envs = 4  # Number of parallel environments

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
