import random
import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class FreewayEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(FreewayEnv, self).__init__()
        pygame.init()
        self.window_width = 800
        self.window_height = 600
        self.player_width = 30
        self.player_height = 30
        self.car_width = 50
        self.car_height = 50

        # Define action and observation space
        # Actions: 0 - Stay, 1 - Move Up, 2 - Move Down
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(self.window_height, self.window_width, 3),
                                            dtype=np.uint8)

        # Load images
        self.window = pygame.display.set_mode((self.window_width, self.window_height))

        self.background_image = pygame.transform.scale(pygame.image.load("../../images/Atari - background.png"), (self.window_width, self.window_height))
        self.player_image = pygame.transform.scale(pygame.image.load("../../images/chicken.png").convert_alpha(), (self.player_width, self.player_height))
        self.car_image = pygame.transform.scale(pygame.image.load("../../images/car2.png").convert_alpha(), (self.car_width, self.car_height))

        self.clock = pygame.time.Clock()

        self.reset()

    def reset(self):
        self.player_rect = pygame.Rect(self.window_width // 2 - self.player_width // 2,
                                       self.window_height - self.player_height - 10,
                                       self.player_width, self.player_height)
        self.score = 0
        self.cars = [{'x': random.randint(0, self.window_width - self.car_width),
                      'lane': random.choice([100, 200, 300, 400, 500, 600, 700]),
                      'speed': random.randint(2, 5)} for _ in range(20)]
        self.done = False
        self.episode_start_time = pygame.time.get_ticks()
        return self.get_observation()

    def step(self, action):
        if action == 1:  # Up
            self.player_rect.y = max(0, self.player_rect.y - 5)
        elif action == 2:  # Down
            self.player_rect.y = min(self.window_height - self.player_height, self.player_rect.y + 5)

        for car in self.cars:
            car['x'] += car['speed']
            if car['x'] > self.window_width:
                car['x'] = -random.randint(100, 300)
                car['speed'] = random.randint(2, 5)

        # Collision detection
        hit = any(self.player_rect.colliderect(pygame.Rect(car['x'], car['lane'], self.car_width, self.car_height)) for car in self.cars)
        if hit:
            self.score = 0
            # reset the player position
            self.player_rect.y = self.window_height - self.player_height - 10

        current_time = pygame.time.get_ticks()
        if current_time - self.episode_start_time >= 60000:  # 60000 milliseconds = 1 minute
            self.done = True
            
        if self.player_rect.y <= 0:  # Reached top
            self.score += 1
            self.player_rect.y = self.window_height - self.player_height - 10

        return self.get_observation(), self.score, self.done, {}

    def render(self, mode='human'):
        self.window.blit(self.background_image, (0, 0))
        for car in self.cars:
            self.window.blit(self.car_image, (car['x'], car['lane']))
        self.window.blit(self.player_image, (self.player_rect.x, self.player_rect.y))
        pygame.display.update()

    def get_observation(self):
        # Optionally return a screenshot of the game as an observation
        # You can also choose to return other representations of the game state
        return np.array(pygame.surfarray.array3d(pygame.display.get_surface()))

    def close(self):
        pygame.quit()

# Example usage
if __name__ == "__main__":
    env = FreewayEnv()
    env.reset()

    done = False
    try:
        while not done:
            action = env.action_space.sample()
            _, _, done, _ = env.step(action)
            env.render()
            pygame.time.wait(10)
    finally:
        print(done)
        env.close() 

