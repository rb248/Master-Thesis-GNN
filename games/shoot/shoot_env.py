import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import SubprocVecEnv
import pygame
import random
import numpy as np
from collections import deque
from skimage.color import rgb2gray
from skimage.transform import resize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO

class ShootingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, observation_type='pixel', max_steps=50, frame_stack=4):
        super(ShootingEnv, self).__init__()
        
        self.observation_type = observation_type
        self.max_steps = max_steps
        self.frame_stack = frame_stack
        self.current_step = 0

        # Game window dimensions
        self.WIDTH, self.HEIGHT = 800, 600

        # Colors
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.BLACK = (0, 0, 0)

        # Shooter settings
        self.shooter_pos = [self.WIDTH // 2, self.HEIGHT - 50]
        self.shooter_speed = 5
        
        # Target settings
        self.target_pos = [random.randint(20, self.WIDTH - 20), random.randint(20, self.HEIGHT // 2)]
        self.target_speed = 4
        self.target_direction = 1

        # Bullet settings
        self.bullet_pos = None  # No bullet initially
        self.bullet_speed = 10

        # Gym spaces
        self.action_space = spaces.Discrete(2)  # 0: do nothing, 1: shoot bullet

        if observation_type == 'pixel':
            self.observation_space = spaces.Box(low=0, high=1, shape=(self.frame_stack, 84, 84), dtype=np.float32)
        else:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2, 4), dtype=np.float32)

        self.frame_buffer = deque(maxlen=self.frame_stack)

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        
        if not hasattr(self, 'win'):
            pygame.init()
            self.win = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("Shooting Game")
            self.offscreen_surface = pygame.Surface((self.WIDTH, self.HEIGHT))
            self.clock = pygame.time.Clock()
        
        if seed is not None:
            self.seed(seed)  # Seed the RNG for the environment
        self.shooter_pos = [self.WIDTH // 2, self.HEIGHT - 50]
        self.target_pos = [random.randint(20, self.WIDTH - 20), random.randint(20, self.HEIGHT // 2)]
        self.bullet_pos = None  # Reset bullet position
        self.target_direction = 1
        self.current_step = 0
        self.frame_buffer.clear()
        if self.observation_type == 'pixel':
            obs = self._get_obs()
            for _ in range(self.frame_stack):
                self.frame_buffer.append(obs)
            
            return np.array(self.frame_buffer), {}
        else:
            return self._get_object_data(), {}

    def step(self, action):
        reward = 0
        done = False
        
        # Shoot bullet if action is 1 and there's no bullet currently
        if action == 1 and self.bullet_pos is None:
            self._shoot_bullet()
        
        # Move target
        self._move_target()
        
        # Move bullet
        self._move_bullet()

        # Check collision or miss
        reward, done = self._check_collision_or_miss()

        # Increment step counter
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True

        # Step penalty to encourage faster completion
        reward -= 0.01

        # Update observation
        obs = self._get_obs()
        self.frame_buffer.append(obs)
        
        return np.array(self.frame_buffer), reward, done, False, {}

    def render(self, mode="human"):
        if mode == "human":
            self._render_on_surface(self.win)
            pygame.display.update()

    def close(self):
        pygame.quit()

    def _get_obs(self):
        self._render_on_surface(self.offscreen_surface)
        obs = pygame.surfarray.array3d(self.offscreen_surface)
        obs = np.transpose(obs, (1, 0, 2))  # Transpose to match the shape (height, width, channels)
        obs = rgb2gray(obs)  # Convert to grayscale
        obs = resize(obs, (84, 84), anti_aliasing=True, preserve_range=True).astype(np.float32) / 255.0  # Normalize to [0, 1]
        return obs

    def _render_on_surface(self, surface):
        surface.fill((0, 0, 0))
        pygame.draw.rect(surface, (255, 255, 255), self.left_paddle)
        pygame.draw.rect(surface, (255, 255, 255), self.right_paddle)
        pygame.draw.ellipse(surface, (255, 255, 255), self.ball)

    def _get_object_data(self):
        objects = []
        
        if self.bullet_pos:
            objects.append([self.bullet_pos[0], self.bullet_pos[1], self.bullet_speed, 1])  # Bullet

        # Append target
        objects.append([self.target_pos[0], self.target_pos[1], self.target_speed * self.target_direction, 2])

        return np.array(objects, dtype=np.float32)

    def _move_target(self):
        self.target_pos[0] += self.target_speed * self.target_direction
        if self.target_pos[0] >= self.WIDTH - 20 or self.target_pos[0] <= 20:
            self.target_direction *= -1

    def _shoot_bullet(self):
        self.bullet_pos = [self.shooter_pos[0] + 20, self.shooter_pos[1]]

    def _move_bullet(self):
        if self.bullet_pos:
            self.bullet_pos[1] -= self.bullet_speed
            if self.bullet_pos[1] < 0:
                self.bullet_pos = None  # Remove the bullet when it goes off-screen

    def _check_collision_or_miss(self):
        reward = -1  # Default to negative reward for missing
        done = False

        if self.bullet_pos:
            if self.target_pos[0] - 20 < self.bullet_pos[0] < self.target_pos[0] + 20 and self.target_pos[1] - 20 < self.bullet_pos[1] < self.target_pos[1] + 20:
                # Bullet hits the target
                reward = 10  # Positive reward for hitting the target
                self.bullet_pos = None  # Remove the bullet after hitting the target
                done = True  # End the episode on hit

        return reward, done

if __name__ == "__main__":
    env = ShootingEnv(observation_type='pixel', max_steps=1000)
    #env = make_vec_env(lambda: ShootingEnv(observation_type='pixel', max_steps=1000), n_envs=4, vec_env_cls=SubprocVecEnv)
    model = PPO.load("ppo_shooting_pixel")
    obs,_ = env.reset()
    done = False
    render_mode = True  # Change this to True if you want to render

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        if render_mode:
            env.render()
        
