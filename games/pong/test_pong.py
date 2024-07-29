import gymnasium as gym
import time

# Create the environment
env = gym.make("PongNoFrameskip-v4", render_mode="human")

# Reset the environment
obs = env.reset()

# Run a few steps to gather data and render the game
n_steps = 1000000
total_reward = 0

for step in range(n_steps):
    # Render the environment
    env.render()
    
    # Take a random action
    action = env.action_space.sample()
    obs, reward, done, _,info = env.step(action)
    
    # Collect rewards
    total_reward += reward
    
    # Check if the environment is done
    if done:
        print(f"Game finished after {step+1} steps with reward {total_reward}")
        total_reward = 0  # Reset the reward for the next game
        obs = env.reset()  # Reset the environment

    # Slow down the rendering
    time.sleep(0.01)

# To check the size of the observation space
print("Observation space shape:", obs.shape)
print("Sample observation:", obs)

# To check the size of the action space
print("Action space:", env.action_space)

# Close the environment
env.close()
