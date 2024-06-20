import random
import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import networkx as nx
from torch_geometric.data import HeteroData, Batch
from collections import defaultdict
from itertools import combinations

class FreewayEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, render_mode='human', observation_type='pixel', frame_stack=4):
        super(FreewayEnv, self).__init__()
        pygame.init()
        self.render_mode = render_mode
        self.observation_type = observation_type
        self.window_width = 800
        self.window_height = 600
        self.player_width = 30
        self.player_height = 30
        self.car_width = 50
        self.car_height = 50
        self.frame_stack = frame_stack

        self.lanes = [100, 200, 300, 400, 500, 600, 700]
        self.max_cars = 20
        # Define action and observation space
        # Actions: 0 - Stay, 1 - Move Up, 2 - Move Down
        self.action_space = spaces.Discrete(3)

        if observation_type == "pixel":
            self.observation_space = spaces.Box(low=0, high=255, shape=(self.frame_stack, 84, 84), dtype=np.uint8)
        else:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(20 + 3, 7), dtype=np.float32)

        self.window = pygame.display.set_mode((self.window_width, self.window_height))
        self.background_image = pygame.image.load("games/images/Atari - background.png")
        self.background_image = pygame.transform.scale(self.background_image, (self.window_width, self.window_height))
        self.player_image = pygame.image.load("games/images/chicken.png").convert_alpha()
        self.player_image = pygame.transform.scale(self.player_image, (self.player_width, self.player_height))
        self.car_image = pygame.image.load("games/images/car2.png").convert_alpha()
        self.car_image = pygame.transform.scale(self.car_image, (self.car_width, self.car_height))
        self.frame_buffer = np.zeros((self.frame_stack, 84, 84), dtype=np.uint8)

        self.clock = pygame.time.Clock()
        self.reset()
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        random.seed(seed)
        np.random.seed(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        if seed is not None:
            self.seed(seed)
        self.player_rect = pygame.Rect(self.window_width // 2 - self.player_width // 2,
                                       self.window_height - self.player_height - 10,
                                       self.player_width, self.player_height)
        self.score = 0
        self.cars = [{'x': random.randint(0, self.window_width - self.car_width),
                      'lane': random.choice(self.lanes),
                      'speed': random.randint(2, 5)} for _ in range(20)]
        self.done = False
        self.episode_start_time = pygame.time.get_ticks()
        self.frame_buffer = np.zeros((self.frame_stack, 84, 84), dtype=np.uint8)
        if self.observation_type == "pixel":
            for _ in range(self.frame_stack):
                self.update_frame_buffer()
            return self.get_observation(), {}
        else:
            return self.get_object_data(), {}

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
            self.score = -1
            self.player_rect.y = self.window_height - self.player_height - 10

        current_time = pygame.time.get_ticks()
        if current_time - self.episode_start_time >= 60000:  # 60000 milliseconds = 1 minute
            self.done = True
            
        if self.player_rect.y <= 0:  # Reached top
            self.score = 10
            self.player_rect.y = self.window_height - self.player_height - 10

        if self.observation_type == "pixel":
            self.update_frame_buffer()
            observation = self.get_observation()
        else:
            observation = self.get_object_data()

        return observation, self.score, self.done, False, {}

    def update_frame_buffer(self):
        frame = self.render_to_array()
        grayscale = np.dot(frame[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)  # Convert to grayscale
        resized_frame = pygame.transform.scale(pygame.surfarray.make_surface(grayscale), (84, 84))
        frame_array = pygame.surfarray.array3d(resized_frame).transpose(1, 0, 2)[:, :, 0]

        self.frame_buffer = np.roll(self.frame_buffer, shift=-1, axis=0)
        self.frame_buffer[-1] = frame_array

    def render_to_array(self):
        self.window.blit(self.background_image, (0, 0))
        for car in self.cars:
            self.window.blit(self.car_image, (car['x'], car['lane']))
        self.window.blit(self.player_image, (self.player_rect.x, self.player_rect.y))
        return pygame.surfarray.array3d(self.window)

    def get_observation(self):
        return self.frame_buffer

    def get_object_data(self):
        objects = [
            [self.player_rect.x, self.player_rect.y, 0, 0, 1, 0, 0],  # Player
            [self.window_width/2, 0, 0, 0, 0, 1, 0],  # Top wall
            [self.window_width/2, self.window_height, 0, 0, 0, 1, 0]  # Bottom wall
        ]

        for i, car in enumerate(self.cars):
            objects.append([car['x'], car['lane'], car['speed'], 0, 0, 0, 1])

        while len(objects) < self.max_cars + 3:  # Ensure the list has a constant length
            objects.append([0, 0, 0, 0, 0, 0, 0])

        return torch.tensor(objects, dtype=torch.float32)

    def render(self, mode='human'):
        self.window.blit(self.background_image, (0, 0))
        for car in self.cars:
            self.window.blit(self.car_image, (car['x'], car['lane']))
        self.window.blit(self.player_image, (self.player_rect.x, self.player_rect.y))
        pygame.display.update()

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    env = FreewayEnv(render_mode='human', observation_type='pixel')
    env.reset()

    done = False
    try:
        while not done:
            action = env.action_space.sample()
            _, _, done, _,_ = env.step(action)
            env.render()
            pygame.time.wait(10)
    finally:
        env.close()
