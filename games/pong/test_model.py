from games.pong.pong_envs.pong_env import PongEnvNew
from stable_baselines3 import PPO

# Create a new environment for testing
test_env = PongEnvNew(render_mode='human', observation_type='pixel')
# load the model
model = PPO.load("ppo_custom_cnn_pong")
# Initialize the state of the environment
obs = test_env.reset()

# Run the model on the test environment
for _ in range(1000):  # Adjust the range as needed
    action, _states = model.predict(obs)
    obs, rewards, dones, info = test_env.step(action)
    test_env.render()

# Close the environment after testing
test_env.close()