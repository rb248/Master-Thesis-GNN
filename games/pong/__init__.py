from gym.envs.registration import register

register(
    id="pong_envs/Pong-v0",
    entry_point="pong_envs:GridWorldEnv",
)
