from games.pong.pong_envs.pong_env import PongEnvNew
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

def optimize_ppo(trial):
    # Define the hyperparameters to tune
    n_steps = trial.suggest_int('n_steps', 2048, 8192)
    gamma = trial.suggest_float('gamma', 0.8, 0.9999, log=True)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    ent_coef = trial.suggest_float('ent_coef', 0.00001, 0.1, log=True)
    clip_range = trial.suggest_float('clip_range', 0.1, 0.4)
    n_epochs = trial.suggest_int('n_epochs', 1, 10)
    batch_size = trial.suggest_int('batch_size', 32, 256)

    # Create the vectorized environment
    env = make_vec_env(lambda: PongEnvNew(render_mode=None, observation_type="pixel"), n_envs=4)

    # Create the PPO model
    model = PPO("CnnPolicy", env, n_steps=n_steps, gamma=gamma, learning_rate=learning_rate, ent_coef=ent_coef,
                clip_range=clip_range, n_epochs=n_epochs, batch_size=batch_size, verbose=0)
    
    # Train the model
    model.learn(total_timesteps=50000)  # Initial exploration phase

    # Evaluate the model
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)

    return mean_reward 


study = optuna.create_study(direction='maximize')
study.optimize(optimize_ppo, n_trials=20)

def optimize_ppo_refined(trial):
    # Use the best parameters from the initial phase
    best_params = study.best_params
    n_steps = trial.suggest_int('n_steps', best_params['n_steps'] - 1000, best_params['n_steps'] + 1000)
    gamma = trial.suggest_float('gamma', best_params['gamma'] - 0.05, best_params['gamma'] + 0.05)
    learning_rate = trial.suggest_float('learning_rate', best_params['learning_rate'] - 1e-5, best_params['learning_rate'] + 1e-5)
    ent_coef = trial.suggest_float('ent_coef', best_params['ent_coef'] - 0.01, best_params['ent_coef'] + 0.01)
    clip_range = trial.suggest_float('clip_range', best_params['clip_range'] - 0.1, best_params['clip_range'] + 0.1)
    n_epochs = trial.suggest_int('n_epochs', best_params['n_epochs'] - 2, best_params['n_epochs'] + 2)
    batch_size = trial.suggest_int('batch_size', best_params['batch_size'] - 32, best_params['batch_size'] + 32)

    # Create the vectorized environment
    env = make_vec_env(lambda: PongEnvNew(), n_envs=4)

    # Create the PPO model
    model = PPO("CnnPolicy", env, n_steps=n_steps, gamma=gamma, learning_rate=learning_rate, ent_coef=ent_coef,
                clip_range=clip_range, n_epochs=n_epochs, batch_size=batch_size, verbose=0)
    
    # Train the model with more timesteps
    model.learn(total_timesteps=200000)  # Refinement phase

    # Evaluate the model
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)

    return mean_reward

# Run the refinement phase
study.optimize(optimize_ppo_refined, n_trials=20)

# Print the best hyperparameters
print(study.best_params)
