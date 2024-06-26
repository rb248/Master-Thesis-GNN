import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from wandb.integration.sb3 import WandbCallback
#from games.model.policy import CustomActorCriticPolicy
from games.breakout.breakout_env import BreakoutEnv
from games.model.policy import CustomCNN, CustomHeteroGNN
import pygame
#Initialize wandb
wandb.init(
    project="gnn_atari_breakout",  # Replace with your project name
    sync_tensorboard=True,        # Automatically sync SB3 logs with wandb
    monitor_gym=True,             # Automatically log gym environments
    save_code=True                # Save the code used for this run
)



env = BreakoutEnv(render_mode='rgb_array', observation_type='graph', training=True)
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
        arity_dict={'atom': 2},
        game = 'breakout'
    ),
)

# # Create the PPO model with the custom feature extractor
model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=2)
# # Train the model with WandbCallback
model.learn(total_timesteps=1000000, callback=WandbCallback())
# # Save the model
model.save("ppo_breakout_heterognn")

