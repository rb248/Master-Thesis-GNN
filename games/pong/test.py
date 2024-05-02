import gym

env = gym.make('Pong-v0')
env.reset()
for _ in range(1000):
    env.render()
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    if done:
        env.reset()
env.close()