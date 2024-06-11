import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv

class RandomPolicy:
    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, observation, deterministic=False):
        action = self.action_space.sample()
        return [action], None  # Return a list of actions to match the vectorized environment's expectation

# Create the Pong environment
env = gym.make('ALE/Pong-v5', render_mode='human')
env = DummyVecEnv([lambda: env])

# Initialize the random policy
random_policy = RandomPolicy(env.action_space)

# Run the random agent in the environment
obs = env.reset()
for _ in range(1000):  # Run for 1000 steps
    action, _ = random_policy.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

# Close the environment
env.close()
