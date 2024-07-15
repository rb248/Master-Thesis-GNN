import random
import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
from collections import deque
from skimage.color import rgb2gray
from skimage.transform import resize

class FreewayEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, render_mode='human', observation_type='pixel', frame_stack=4, lanes=[50, 80, 120], max_cars=20, car_speed=1):
        super(FreewayEnv, self).__init__()
        pygame.init()
        self.last_time = pygame.time.get_ticks()
        self.render_mode = render_mode
        self.observation_type = observation_type
        self.window_width = 210
        self.window_height = 160
        self.player_width = 5
        self.player_height = 5
        self.car_width = 20
        self.car_height = 20
        self.frame_stack = frame_stack
        self.lanes = lanes
        self.max_cars_init = max_cars
        self.car_speed = car_speed

        self.action_space = spaces.Discrete(3)

        if observation_type == "pixel":
            self.observation_space = spaces.Box(low=0, high=255, shape=(self.frame_stack, 84, 84), dtype=np.uint8)
        else:
            self.max_objects = self.max_cars_init + 4
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_objects, 7), dtype=np.float32)

        self.window = pygame.display.set_mode((self.window_width, self.window_height))
        self.background_image = pygame.image.load("games/images/Atari - background.png")
        self.background_image = pygame.transform.scale(self.background_image, (self.window_width, self.window_height))
        self.player_image = pygame.image.load("games/images/chicken.png").convert_alpha()
        self.player_image = pygame.transform.scale(self.player_image, (self.player_width, self.player_height))
        self.car_image = pygame.image.load("games/images/car2.png").convert_alpha()
        self.car_image = pygame.transform.scale(self.car_image, (self.car_width, self.car_height))
        self.frame_buffer = deque(maxlen=self.frame_stack)

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
        lane_combinations = [[50,80,120],[50,80],[50,120],[80,120]]
        number_cars = [10,15,20]
        car_speeds = [1,2,3]
        self.lanes = random.choice(lane_combinations)
        self.max_cars = random.choice(number_cars)
        self.car_speed = random.choice(car_speeds)
        self.cars = [{'x': random.randint(0, self.window_width - self.car_width),
                      'lane': random.choice(self.lanes),
                      'speed': self.car_speed} for car in range(self.max_cars)]
        self.done = False
        self.episode_step = 0
        self.frame_buffer.clear()
        if self.observation_type == "pixel":
            for _ in range(self.frame_stack):
                self.frame_buffer.append(self._get_obs())
            return np.array(self.frame_buffer), {}
        else:
            return self.get_object_data(), {}

    def step(self, action):
        reward = -0.5
        current_time = pygame.time.get_ticks()
        if action == 1:  # Up
            self.player_rect.y = max(0, self.player_rect.y - 5)
        elif action == 2:  # Down
            self.player_rect.y = min(self.window_height - self.player_height, self.player_rect.y + 5)

        for car in self.cars:
            car['x'] += car['speed']
            if car['x'] > self.window_width:
                car['x'] = 0
                car['speed'] = self.car_speed

        hit = any(self.player_rect.colliderect(pygame.Rect(car['x'], car['lane'], self.car_width, self.car_height)) for car in self.cars)
        if hit:
            self.player_rect.y = self.window_height - self.player_height - 10
            self.last_time = current_time
        done = False
        self.episode_step += 1
        if self.episode_step >= 3000:
            done = True

        if self.player_rect.y <= 0:  # Reached top
            self.score += 1
            reward += 10 * (len(self.lanes))
            self.player_rect.y = self.window_height - self.player_height - 10
        truncated = False
        info = {}  
        if self.observation_type == "pixel":
            observation = self.get_observation()
            return np.array(self.frame_buffer), reward, done, truncated, info
        else:
            return self.get_object_data(), reward, done, truncated, info



    def _get_obs(self):
        frame = self.render_to_array()
    
        # Convert to grayscale
        grayscale = rgb2gray(frame)
        
        # Normalize the grayscale image to enhance contrast
        normalized_frame = (grayscale - grayscale.min()) / (grayscale.max() - grayscale.min())
        
        # Resize the frame
        resized_frame = resize(normalized_frame, (84, 84), anti_aliasing=True, mode='reflect', preserve_range=True)
        
        return resized_frame

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
        ]
        for lane in self.lanes:
            objects.append([self.window_width // 2, lane, 0, 0, 0, 1, 0])

        for car in self.cars:
            objects.append([car['x'], car['lane'], car['speed'], 0, 0, 0, 1])

        while len(objects) < self.max_objects:
            objects.append([0, 0, 0, 0, 0, 0, 0])

        return torch.tensor(objects, dtype=torch.float32)

    def render(self, mode='human'):
        self.window.blit(self.background_image, (0, 0))
        for car in self.cars:
            self.window.blit(self.car_image, (car['x'], car['lane']))
        self.window.blit(self.player_image, (self.player_rect.x, self.player_rect.y))
        pygame.display.update()
        self.clock.tick(60)

    def close(self):
        pygame.quit()


if __name__=="__main__":
    env = FreewayEnv(render_mode='human', observation_type='graph')

    #model = PPO.load("best_model")
    #model = PPO.load("ppo_freeway_pixel")
    #model = PPO.load("ppo_custom_heterognn")

    # # Evaluate the agent
    # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1, render=True)
    # print(f"Mean reward: {mean_reward} ± {std_reward}")

    obs,_ = env.reset()
    done = False
    total_reward = 0
    while not done:
        #action, _ = model.predict(obs)
        action = env.action_space.sample()
        obs, reward, done, _,_ = env.step(action)
        total_reward += reward
        #print(f"Action: {action}, Reward: {reward}")
        pygame.time.delay(50)
        env.render()

    print(f"Total reward: {total_reward}")