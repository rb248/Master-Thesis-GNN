from stable_baselines3.common.env_checker import check_env
from pong_env import PongEnv


env = PongEnv()
# It will check your custom environment and output additional warnings if needed
check_env(env)