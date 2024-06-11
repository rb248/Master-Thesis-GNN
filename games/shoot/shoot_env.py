import gym
from gym import spaces
import pygame
import random
import numpy as np
import time

class ShootingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self):
        super(ShootingEnv, self).__init__()
        pygame.init()
        
        # Game window dimensions
        self.WIDTH, self.HEIGHT = 800, 600
        self.win = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Shooting Game")

        # Colors
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.BLACK = (0, 0, 0)

        # Shooter settings
        self.shooter_pos = [self.WIDTH // 2, self.HEIGHT - 50]
        self.shooter_speed = 5

        # Target settings
        self.target_pos = [random.randint(20, self.WIDTH - 20), random.randint(20, self.HEIGHT // 2)]
        self.target_speed = 2
        self.target_direction = 1

        # Bullet settings
        self.bullet_pos = []
        self.bullet_speed = 10

        # Gym spaces
        self.action_space = spaces.Discrete(2)  # 0: do nothing, 1: shoot bullet
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)

        self.clock = pygame.time.Clock()

    def reset(self):
        self.shooter_pos = [self.WIDTH // 2, self.HEIGHT - 50]
        self.target_pos = [random.randint(20, self.WIDTH - 20), random.randint(20, self.HEIGHT // 2)]
        self.bullet_pos = []
        self.target_direction = 1
        return self._get_obs()

    def step(self, action):
        reward = 0
        done = False
        
        # Shoot bullet if action is 1
        if action == 1:
            self._shoot_bullet()
        
        # Move target
        self._move_target()
        
        # Move bullets
        self._move_bullets()

        # Check collision
        reward, done = self._check_collision()

        obs = self._get_obs()
        return obs, reward, done, {}

    def render(self, mode="human"):
        self.win.fill(self.BLACK)
        pygame.draw.circle(self.win, self.RED, self.target_pos, 20)
        pygame.draw.rect(self.win, self.WHITE, (*self.shooter_pos, 50, 20))
        for bullet in self.bullet_pos:
            pygame.draw.rect(self.win, self.WHITE, (*bullet, 10, 5))
        pygame.display.update()

    def close(self):
        pygame.quit()

    def _get_obs(self):
        self.render()  # Render to the pygame display
        obs = pygame.surfarray.array3d(pygame.display.get_surface())
        return np.transpose(obs, (1, 0, 2))  # Transpose to match the shape (height, width, channels)

    def _move_target(self):
        self.target_pos[0] += self.target_speed * self.target_direction
        if self.target_pos[0] >= self.WIDTH - 20 or self.target_pos[0] <= 20:
            self.target_direction *= -1

    def _shoot_bullet(self):
        self.bullet_pos.append([self.shooter_pos[0] + 20, self.shooter_pos[1]])

    def _move_bullets(self):
        for bullet in self.bullet_pos[:]:
            bullet[1] -= self.bullet_speed
            if bullet[1] < 0:
                self.bullet_pos.remove(bullet)

    def _check_collision(self):
        reward = 0
        done = False
        for bullet in self.bullet_pos:
            if self.target_pos[0] - 20 < bullet[0] < self.target_pos[0] + 20 and self.target_pos[1] - 20 < bullet[1] < self.target_pos[1] + 20:
                self.shooter_pos = [self.WIDTH // 2, self.HEIGHT - 50]
                self.target_pos = [random.randint(20, self.WIDTH - 20), random.randint(20, self.HEIGHT // 2)]
                self.bullet_pos.remove(bullet)
                reward = 1
                done = True
            else:
                reward = 0
        return reward, done

# Example usage
if __name__ == "__main__":
    env = ShootingEnv()
    obs = env.reset()

    done = False
    while not done:
        action = env.action_space.sample()  # Random action for illustration
        obs, reward, done, info = env.step(action)
        time.sleep(0.1)
        env.render()

    env.close()
