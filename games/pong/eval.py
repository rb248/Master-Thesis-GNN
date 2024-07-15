from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from games.pong.pong_envs.pong_env import PongEnvNew
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
import numpy as np


def evaluate_model_on_configs(model_path, env_configs, n_eval_episodes=10):
    # Load the model
    model = PPO.load(model_path)

    results = []

    for config in env_configs:
        print(f"Evaluating with config: {config}")

        # Create a single instance of the environment for evaluation with the given config
        eval_env = PongEnvNew(**config)

        # Evaluate the policy
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=n_eval_episodes)

        results.append({
            "config": config,
            "mean_reward": mean_reward,
            "std_reward": std_reward
        })

        print(f"Mean reward: {mean_reward} ± {std_reward}")

    return results

if __name__ == "__main__":
    # num_envs = 4  # Number of parallel environments

    # # Create a vectorized environment with DummyVecEnv
    # def make_env(rank, seed=0):
    #     def _init():
    #         env = PongEnvNew(render_mode='rgb_array', observation_type='pixel')
    #         env.seed(seed + rank)
    #         return env
    #     return _init

    # env = DummyVecEnv([make_env(i) for i in range(num_envs)])

    

    # Save the model

    # Define environment configurations for evaluation
    env_configs = [
        {"render_mode": "human", "observation_type": "pixel", "paddle_width": 5, "ball_speed": 4},
        {"render_mode": "human", "observation_type": "pixel", "paddle_width": 5, "ball_speed": 3},
        {"render_mode": "human", "observation_type": "pixel", "paddle_width": 5, "ball_speed": 2},
        {"render_mode": "human", "observation_type": "pixel", "paddle_width": 5, "ball_speed": 1}
    ]

    # Evaluate the model on different configurations
    results = evaluate_model_on_configs("ppo_pong_pixel", env_configs)

    for result in results:
        config = result["config"]
        mean_reward = result["mean_reward"]
        std_reward = result["std_reward"]
        print(f"Config: {config} - Mean reward: {mean_reward} ± {std_reward}")