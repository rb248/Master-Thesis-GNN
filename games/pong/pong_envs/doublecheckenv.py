from pong_env import PongEnv


env = PongEnv()
episodes = 100

for episode in range(episodes):
	done = False
	obs = env.reset()
	while True:#not done:
		random_action = env.action_space.sample()
		print("action",random_action)
		obs, reward, done, truncated, info = env.step(random_action)
		print('reward',reward)