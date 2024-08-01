from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from games.pong.pong_envs.pong_env import PongEnvTest
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
import matplotlib.pyplot as plt

def custom_evaluate_policy(model, env, n_eval_episodes=10, render=False):
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(n_eval_episodes):
        obs,_ = env.reset()
        done = False
        total_rewards = 0
        steps = 0
        total_rewards = 0
        while not done:
            action,_ = model.predict(obs)
            #action = env.action_space.sample()
            obs, reward, done, _, _ = env.step(action)
            total_rewards += reward 
            if render:
                env.render()
            steps += 1
            if steps > 5000:    
                break
        print(f"Episode {episode + 1} - Total reward: {total_rewards}")
        print(f"Episode {episode + 1} - Total steps: {steps}")
        episode_rewards.append(total_rewards)
        episode_lengths.append(steps)
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    
    return mean_reward, std_reward, mean_length


def evaluate_model_on_configs(model_path, env_configs, n_eval_episodes=10):
    # Load the model
    model = PPO.load(model_path)  

    results = []

    for config in env_configs:
        print(f"Evaluating with config: {config}")

        # Create a single instance of the environment for evaluation with the given config
        eval_env = PongEnvTest(**config)

        # Evaluate the policy
        mean_reward, std_reward, _ = custom_evaluate_policy(model, eval_env, n_eval_episodes=n_eval_episodes)

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
    # env_configs = [
    #     {"render_mode": None, "observation_type": "graph", "paddle_width": 5, "ball_speed": 4},
    #     {"render_mode": None, "observation_type": "graph", "paddle_width": 4, "ball_speed":4},
    #     {"render_mode": None, "observation_type": "graph", "paddle_width": 3, "ball_speed": 4},
    #     {"render_mode": None, "observation_type": "graph", "paddle_width": 2, "ball_speed": 4}
    # ]
    # env_configs = [

    #     {"render_mode": None, "observation_type": "pixel", "paddle_width": 6, "ball_speed": 4},
    #     {"render_mode": None, "observation_type": "pixel", "paddle_width": 5, "ball_speed": 4},
    #     {"render_mode": None, "observation_type": "pixel", "paddle_width": 4, "ball_speed": 4},
    #     {"render_mode": None, "observation_type": "pixel", "paddle_width": 3, "ball_speed": 4},
    #     {"render_mode": None, "observation_type": "pixel", "paddle_width": 2, "ball_speed": 4}

    # ]
    # 
    # env_configs = [

    #     {"render_mode": None, "observation_type": "pixel", "paddle_width": 1, "ball_speed": 4},
    #     {"render_mode": None, "observation_type": "pixel", "paddle_width": 2, "ball_speed": 4},
    #     {"render_mode": None, "observation_type": "pixel", "paddle_width": 3, "ball_speed": 4},
    #     {"render_mode": None, "observation_type": "pixel", "paddle_width": 4, "ball_speed": 4},
    #     {"render_mode": None, "observation_type": "pixel", "paddle_width": 5, "ball_speed": 4},
    #     {"render_mode": None, "observation_type": "pixel", "paddle_width": 6, "ball_speed": 4},
    #     {"render_mode": None, "observation_type": "pixel", "paddle_width": 7, "ball_speed": 4},
    #     {"render_mode": None, "observation_type": "pixel", "paddle_width": 8, "ball_speed": 4},
    #     {"render_mode": None, "observation_type": "pixel", "paddle_width": 9, "ball_speed": 4},
    #     {"render_mode": None, "observation_type": "pixel", "paddle_width": 10, "ball_speed": 4}
    # ] 
    # env_configs = [

    #     {"render_mode": None, "observation_type": "graph", "paddle_width": 1, "ball_speed": 4},
    #     {"render_mode": None, "observation_type": "graph", "paddle_width": 2, "ball_speed": 4},
    #     {"render_mode": None, "observation_type": "graph", "paddle_width": 3, "ball_speed": 4},
    #     {"render_mode": None, "observation_type": "graph", "paddle_width": 4, "ball_speed": 4},
    #     {"render_mode": None, "observation_type": "graph", "paddle_width": 5, "ball_speed": 4},
    #     {"render_mode": None, "observation_type": "graph", "paddle_width": 6, "ball_speed": 4},
    #     {"render_mode": None, "observation_type": "graph", "paddle_width": 7, "ball_speed": 4},
    #     {"render_mode": None, "observation_type": "graph", "paddle_width": 8, "ball_speed": 4},
    #     {"render_mode": None, "observation_type": "graph", "paddle_width": 9, "ball_speed": 4},
    #     {"render_mode": None, "observation_type": "graph", "paddle_width": 10, "ball_speed": 4}
    # ]

    env_configs = [

        {"render_mode": None, "observation_type": "graph", "paddle_width": 5, "ball_speed": 10},
        {"render_mode": None, "observation_type": "graph", "paddle_width": 5, "ball_speed": 9},
        {"render_mode": None, "observation_type": "graph", "paddle_width": 5, "ball_speed": 8},
        {"render_mode": None, "observation_type": "graph", "paddle_width": 5, "ball_speed": 7},
        {"render_mode": None, "observation_type": "graph", "paddle_width": 5, "ball_speed": 6},
        {"render_mode": None, "observation_type": "graph", "paddle_width": 5, "ball_speed": 5},
        {"render_mode": None, "observation_type": "graph", "paddle_width": 5, "ball_speed": 4},
        {"render_mode": None, "observation_type": "graph", "paddle_width": 5, "ball_speed":3},
        {"render_mode": None, "observation_type": "graph", "paddle_width": 5, "ball_speed": 2},
        {"render_mode": None, "observation_type": "graph", "paddle_width": 5, "ball_speed": 1}
    ]
    # env_configs = [
    #     {"render_mode": None, "observation_type":"pixel", "paddle_width": 5, "ball_speed": 1},
    #     {"render_mode": None, "observation_type":"pixel", "paddle_width": 5, "ball_speed": 2},
    #     {"render_mode": None, "observation_type":"pixel", "paddle_width": 5, "ball_speed": 3},
    #     {"render_mode": None, "observation_type":"pixel", "paddle_width": 5, "ball_speed": 4},
    #     {"render_mode": None, "observation_type":"pixel", "paddle_width": 5, "ball_speed": 5},
    #     {"render_mode": None, "observation_type":"pixel", "paddle_width": 5, "ball_speed": 6},
    #     {"render_mode": None, "observation_type":"pixel", "paddle_width": 5, "ball_speed": 7},
    #     {"render_mode": None, "observation_type":"pixel", "paddle_width": 5, "ball_speed": 8},
    #     {"render_mode": None, "observation_type":"pixel", "paddle_width": 5, "ball_speed": 9},
    #     {"render_mode": None, "observation_type":"pixel", "paddle_width": 5, "ball_speed": 10}
    # ]
    # env_configs = [
    #     {"render_mode": None, "observation_type": "graph", "paddle_width": 5, "ball_speed": 6},
    #     {"render_mode": None, "observation_type": "graph", "paddle_width": 5, "ball_speed": 5},
    #     {"render_mode": None, "observation_type": "graph", "paddle_width": 5, "ball_speed": 4},
    #     {"render_mode": None, "observation_type": "graph", "paddle_width": 5, "ball_speed":3},
    #     {"render_mode": None, "observation_type": "graph", "paddle_width": 5, "ball_speed": 2},
    #     {"render_mode": None, "observation_type": "graph", "paddle_width": 5, "ball_speed": 1}
    # ]
    #Evaluate the model on different configurations
    #results = evaluate_model_on_configs("ppo_custom_heterognn_1", env_configs)
    #results = evaluate_model_on_configs("PONG-GNN-training-random.zip", env_configs)
    #results = evaluate_model_on_configs("pong-CNN-trainig.zip", env_configs)
    #results = evaluate_model_on_configs("ppo_pong_pixel.zip", env_configs)
    results = evaluate_model_on_configs("pong_gnn_training.zip", env_configs)
    for result in results:
        config = result["config"]
        mean_reward = result["mean_reward"]
        std_reward = result["std_reward"]
        print(f"Config: {config} - Mean reward: {mean_reward} ± {std_reward}")
        # plot the results against the ball_speed with error bars
        plt.errorbar(config["ball_speed"], mean_reward, yerr=std_reward, fmt='ro', capsize=5)

    plt.xlabel('Ball Speed')
    plt.ylabel('Mean Reward')
    plt.title('Mean Reward vs Ball Speed with Standard Deviation')
    plt.show()