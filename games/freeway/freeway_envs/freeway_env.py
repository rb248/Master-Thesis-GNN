# import random
# import pygame
# import numpy as np
# import gymnasium as gym
# from gymnasium import spaces
# import torch
# import networkx as nx
# from torch_geometric.data import HeteroData, Batch
# from collections import defaultdict
# from itertools import combinations
# from stable_baselines3 import PPO
# from stable_baselines3.common.evaluation import evaluate_policy

# class FreewayEnv(gym.Env):
#     metadata = {'render_modes': ['human', 'rgb_array']}

#     def __init__(self, render_mode='human', observation_type='pixel', frame_stack=4):
#         super(FreewayEnv, self).__init__()
#         pygame.init()
#         self.last_time = pygame.time.get_ticks()
#         self.render_mode = render_mode
#         self.observation_type = observation_type
#         #self.window_width = 800
#         self.window_width = 210
#         #self.window_height = 600
#         self.window_height = 160
#         self.player_width = 5
#         self.player_height = 5
#         self.car_width = 20
#         self.car_height = 20
#         self.frame_stack = frame_stack

#         self.lanes = [100, 200, 300, 400, 500, 600, 700]
#         self.lanes = [50,100]
#         self.max_cars = 5
#         # Define action and observation space
#         # Actions: 0 - Stay, 1 - Move Up, 2 - Move Down
#         self.action_space = spaces.Discrete(3)

#         if observation_type == "pixel":
#             self.observation_space = spaces.Box(low=0, high=255, shape=(self.frame_stack, 84, 84), dtype=np.uint8)
#         else:
#             self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_cars+ len(self.lanes)+1, 7), dtype=np.float32)

#         self.window = pygame.display.set_mode((self.window_width, self.window_height))
#         self.background_image = pygame.image.load("games/images/Atari - background.png")
#         self.background_image = pygame.transform.scale(self.background_image, (self.window_width, self.window_height))
#         self.player_image = pygame.image.load("games/images/chicken.png").convert_alpha()
#         self.player_image = pygame.transform.scale(self.player_image, (self.player_width, self.player_height))
#         self.car_image = pygame.image.load("games/images/car2.png").convert_alpha()
#         self.car_image = pygame.transform.scale(self.car_image, (self.car_width, self.car_height))
#         self.frame_buffer = np.zeros((self.frame_stack, 84, 84), dtype=np.uint8)

#         self.clock = pygame.time.Clock()
#         self.reset()
#     def seed(self, seed=None):
#         self.np_random, seed = gym.utils.seeding.np_random(seed)
#         random.seed(seed)
#         np.random.seed(seed)
#         return [seed]

#     def reset(self, seed=None, options=None):
#         super().reset(seed=seed, options=options)
#         if seed is not None:
#             self.seed(seed)
#         self.player_rect = pygame.Rect(self.window_width // 2 - self.player_width // 2,
#                                        self.window_height - self.player_height - 10,
#                                        self.player_width, self.player_height)
#         self.score = 0
#         self.cars = [{'x': random.randint(0, self.window_width - self.car_width),
#                       'lane': random.choice(self.lanes),
#                       'speed': random.randint(1, 2)} for _ in range(self.max_cars)]
#         self.done = False
#         self.episode_start_time = pygame.time.get_ticks()
#         self.frame_buffer = np.zeros((self.frame_stack, 84, 84), dtype=np.uint8)
#         if self.observation_type == "pixel":
#             for _ in range(self.frame_stack):
#                 self.update_frame_buffer()
#             return self.get_observation(), {}
#         else:
#             return self.get_object_data(), {}

#     def step(self, action):
#         reward = 0
#         reward = -0.5
#         current_time = pygame.time.get_ticks()
#         if action == 1:  # Up
#             self.player_rect.y = max(0, self.player_rect.y - 5)
#         elif action == 2:  # Down
#             self.player_rect.y = min(self.window_height - self.player_height, self.player_rect.y + 5)

#         for car in self.cars:
#             car['x'] += car['speed']
#             if car['x'] > self.window_width:
#                 car['x'] = -random.randint(100, 300)
#                 car['speed'] = random.randint(1,2)

#         # Collision detection
#         hit = any(self.player_rect.colliderect(pygame.Rect(car['x'], car['lane'], self.car_width, self.car_height)) for car in self.cars)
#         if hit:
#             #self.score = -1
#             self.player_rect.y = self.window_height - self.player_height - 10
        
#             self.last_time = current_time
#         if current_time - self.episode_start_time >= 60000:  # 60000 milliseconds = 1 minute
#             self.done = True
            
#         if self.player_rect.y <= 0:  # Reached top
#             self.score +=1
#             reward += 10*(len(self.lanes))

#             self.player_rect.y = self.window_height - self.player_height - 10

#         if self.observation_type == "pixel":
#             self.update_frame_buffer()
#             observation = self.get_observation()
#         else:
#             observation = self.get_object_data()

#         return observation, reward, self.done, False, {}

#     def update_frame_buffer(self):
#         frame = self.render_to_array()
#         grayscale = np.dot(frame[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)  # Convert to grayscale
#         resized_frame = pygame.transform.scale(pygame.surfarray.make_surface(grayscale), (84, 84))
#         frame_array = pygame.surfarray.array3d(resized_frame).transpose(1, 0, 2)[:, :, 0]

#         self.frame_buffer = np.roll(self.frame_buffer, shift=-1, axis=0)
#         self.frame_buffer[-1] = frame_array

#     def render_to_array(self):
#         self.window.blit(self.background_image, (0, 0))
#         for car in self.cars:
#             self.window.blit(self.car_image, (car['x'], car['lane']))
#         self.window.blit(self.player_image, (self.player_rect.x, self.player_rect.y))
#         return pygame.surfarray.array3d(self.window)

#     def get_observation(self):
#         return self.frame_buffer

#     def get_object_data(self):
#         objects = [
#             [self.player_rect.x, self.player_rect.y, 0, 0, 1, 0, 0],  # Player
            
#         ] 
#         # add lanes
#         for lane in self.lanes:
#             objects.append([self.window_width//2, lane, 0, 0, 0, 1, 0])

#         for i, car in enumerate(self.cars):
#             objects.append([car['x'], car['lane'], car['speed'], 0, 0, 0, 1])

#         # while len(objects) < self.max_cars + 10:  # Ensure the list has a constant length
#         #     objects.append([0, 0, 0, 0, 0, 0, 0])

#         return torch.tensor(objects, dtype=torch.float32)

#     def render(self, mode='human'):
#         self.window.blit(self.background_image, (0, 0))
#         for car in self.cars:
#             self.window.blit(self.car_image, (car['x'], car['lane']))
#         self.window.blit(self.player_image, (self.player_rect.x, self.player_rect.y))
#         pygame.display.update()

#     def close(self):
#         pygame.quit()


# if __name__=="__main__":
#     env = FreewayEnv(render_mode='human', observation_type='pixel')

#     model = PPO.load("ppo_freeway_pixel")
#     #model = PPO.load("ppo_custom_heterognn")

#     # # Evaluate the agent
#     # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1, render=True)
#     # print(f"Mean reward: {mean_reward} ± {std_reward}")

#     obs,_ = env.reset()
#     done = False
#     total_reward = 0
#     while not done:
#         action, _ = model.predict(obs)
#         #action = env.action_space.sample()
#         obs, reward, done, _,_ = env.step(action)
#         total_reward += reward
#         pygame.time.delay(50)
#         env.render()

#     print(f"Total reward: {total_reward}") 

import random
import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
from collections import deque
from skimage.color import rgb2gray
from skimage.transform import resize
from stable_baselines3 import PPO

class FreewayEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, render_mode='human', observation_type='pixel', frame_stack=4, lanes = [30, 60, 90, 120, 150], max_cars=10, car_speed=1):
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

        self.lanes = lanes
        self.max_cars = max_cars
        self.car_speed = car_speed
        # Define action and observation space
        # Actions: 0 - Stay, 1 - Move Up, 2 - Move Down
        self.action_space = spaces.Discrete(3)

        if observation_type == "pixel":
            self.observation_space = spaces.Box(low=0, high=255, shape=(self.frame_stack, 84, 84), dtype=np.uint8)
        else:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(24, 7), dtype=np.float32)

        self.window = pygame.display.set_mode((self.window_width, self.window_height))
        self.background_image = pygame.image.load("games/images/Atari - background.png")
        self.background_image = pygame.transform.scale(self.background_image, (self.window_width, self.window_height))
        self.player_image = pygame.image.load("games/images/chicken.png").convert_alpha()
        self.player_image = pygame.transform.scale(self.player_image, (self.player_width, self.player_height))
        self.car_image = pygame.image.load("games/images/car.png").convert_alpha()
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
        self.episode_step = 0
        self.player_rect = pygame.Rect(self.window_width // 2 - self.player_width // 2,
                                    self.window_height - self.player_height - 10,
                                    self.player_width, self.player_height)
        self.score = 0

        # Define lane combinations and their weights
        lane_combinations = [
            [30, 60, 90, 120, 150],  # 5-lane setup
            [30, 60, 90, 120],       # 4-lane setup
            [30, 60, 150],           # 3-lane setup
            [30, 90, 120],           # Middle lane setup
            [30, 90, 150],           # 3-lane setup
            [30, 120, 150],          # 3-lane setup
            [60, 90, 120, 150],      # 4-lane setup
            [60, 90, 120],           # 3-lane setup
            [60, 90, 150],           # 3-lane setup
            [60, 120, 150],          # 3-lane setup
            [90, 120, 150]           # Middle lane setup
        ]
        lane_weights = [
            30,  # Higher probability for 5-lane setup
            10,  # Moderate probability for 4-lane setup
            5,   # Lower probability for 3-lane setup
            5,  # Higher probability for middle lane setup
            5,   # Lower probability for 3-lane setup
            5,   # Lower probability for 3-lane setup
            10,  # Moderate probability for 4-lane setup
            5,   # Lower probability for 3-lane setup
            5,   # Lower probability for 3-lane setup
            5,   # Lower probability for 3-lane setup
            10   # Higher probability for middle lane setup
        ]
        # lane_combinations = [[60, 90], [60, 90, 120], [60, 120]]
        # # 
        lane_combinations =[[50,80,120],[50,80],[80,120]]
        #lane_combinations = [[50,80,120]]
        number_cars = [10,12, 15]
        car_speeds = [3,4]

        self.car_speed = random.choice(car_speeds)
        self.player_speed = 0
        #self.lanes = random.choices(lane_combinations, weights=lane_weights, k=1)[0]
        self.lanes = random.choice(lane_combinations)
        self.max_cars = random.choice(number_cars)
        
        #lane_speeds = {30: self.car_speed - 2, 60: self.car_speed - 1, 90: self.car_speed, 120: self.car_speed - 1, 150: self.car_speed - 2}
        lane_speeds = {50: self.car_speed - 2, 80: self.car_speed, 120: self.car_speed-1}
        self.cars = []
        lane_car_count = {lane: 0 for lane in self.lanes}

        for _ in range(self.max_cars):
                lane = random.choice(self.lanes)
                
                if lane_car_count[lane] < 5:
                    car = {'x': random.randint(0, self.window_width - self.car_width), 'lane': lane, 'speed': lane_speeds[lane]}
                    self.cars.append(car)
                    lane_car_count[lane] += 1

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
        reward = 0
        reward = -0.5
        self.episode_step += 1
        if self.episode_step % 3000 == 0:
            self.done = True
        previous_y = self.player_rect.y
        current_time = pygame.time.get_ticks()
        if action == 1 and self.episode_step>=30: # Up
            self.player_rect.y = max(0, self.player_rect.y - 5)
        elif action == 2 and self.episode_step>=30:  # Down
            self.player_rect.y = min(self.window_height - self.player_height, self.player_rect.y + 5)
        self.player_speed = self.player_rect.y - previous_y
        for car in self.cars:
            car['x'] += car['speed']
            if car['x'] > self.window_width:
                # car['x'] = -random.randint(100, 300)
                car['x'] = 0
                #car['speed'] = self.car_speed
                #car['speed'] = self.car_speed 
                # print(f"Car speed: {car['speed']}")
                # print(f"Car lane: {car['lane']}")

        # Collision detection
        hit = any(self.player_rect.colliderect(pygame.Rect(car['x'], car['lane'], self.car_width, self.car_height)) for car in self.cars)
        if hit:
            self.player_rect.y = self.window_height - self.player_height - 10
            #reward -= 3
        
            self.last_time = current_time
        if current_time - self.episode_start_time >= 60000:  # 60000 milliseconds = 1 minute
             self.done = True
            
        if self.player_rect.y <= 0:  # Reached top
            self.score +=1  
            reward += 10*(len(self.lanes))
            


            self.player_rect.y = self.window_height - self.player_height - 10
            self.done = True
            


        if self.observation_type == "pixel":
            self.update_frame_buffer()
            observation = self.get_observation()
        else:
            observation = self.get_object_data()

        return observation, reward, self.done, False, {}

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
            [self.player_rect.x, self.player_rect.y, self.player_speed, 0, 1, 0, 0],  # Player
            
        ] 
        # add lanes
        for lane in self.lanes:
            objects.append([self.window_width//2, lane, 0, 0, 0, 1, 0])

        for i, car in enumerate(self.cars):
            objects.append([car['x'], car['lane'], car['speed'], 0, 0, 0, 1])

        while len(objects) < 24:  # Ensure the list has a constant length
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

class FreewayEnvConstant(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, render_mode='human', observation_type='pixel', frame_stack=4, lanes = [30, 60, 90, 120, 150], max_cars=10, car_speed=1):
        super(FreewayEnvConstant, self).__init__()
        pygame.init()
        self.last_time = pygame.time.get_ticks()
        self.render_mode = render_mode
        self.observation_type = observation_type
        #self.window_width = 800
        self.window_width = 210
        #self.window_height = 600
        self.window_height = 160
        self.player_width = 5
        self.player_height = 5
        self.car_width = 20
        self.car_height = 20
        self.frame_stack = frame_stack

        self.lanes = lanes
        self.max_cars = max_cars
        self.car_speed = car_speed
        # Define action and observation space
        # Actions: 0 - Stay, 1 - Move Up, 2 - Move Down
        self.action_space = spaces.Discrete(3)

        if observation_type == "pixel":
            self.observation_space = spaces.Box(low=0, high=255, shape=(self.frame_stack, 84, 84), dtype=np.uint8)
        else:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(24, 7), dtype=np.float32)

        self.window = pygame.display.set_mode((self.window_width, self.window_height))
        self.background_image = pygame.image.load("games/images/Atari - background.png")
        self.background_image = pygame.transform.scale(self.background_image, (self.window_width, self.window_height))
        self.player_image = pygame.image.load("games/images/chicken.png").convert_alpha()
        self.player_image = pygame.transform.scale(self.player_image, (self.player_width, self.player_height))
        self.car_image = pygame.image.load("games/images/car.png").convert_alpha()
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
        self.episode_step = 0
        self.player_rect = pygame.Rect(self.window_width // 2 - self.player_width // 2,
                                    self.window_height - self.player_height - 10,
                                    self.player_width, self.player_height)
        self.score = 0

        # Define lane combinations and their weights
        lane_combinations = [
            [30, 60, 90, 120, 150],  # 5-lane setup
            [30, 60, 90, 120],       # 4-lane setup
            [30, 60, 150],           # 3-lane setup
            [30, 90, 120],           # Middle lane setup
            [30, 90, 150],           # 3-lane setup
            [30, 120, 150],          # 3-lane setup
            [60, 90, 120, 150],      # 4-lane setup
            [60, 90, 120],           # 3-lane setup
            [60, 90, 150],           # 3-lane setup
            [60, 120, 150],          # 3-lane setup
            [90, 120, 150]           # Middle lane setup
        ]
        lane_weights = [
            30,  # Higher probability for 5-lane setup
            10,  # Moderate probability for 4-lane setup
            5,   # Lower probability for 3-lane setup
            5,  # Higher probability for middle lane setup
            5,   # Lower probability for 3-lane setup
            5,   # Lower probability for 3-lane setup
            10,  # Moderate probability for 4-lane setup
            5,   # Lower probability for 3-lane setup
            5,   # Lower probability for 3-lane setup
            5,   # Lower probability for 3-lane setup
            10   # Higher probability for middle lane setup
        ]
        # lane_combinations = [[60, 90], [60, 90, 120], [60, 120]]
        # # 
        #lane_combinations =[[50,80,120],[50,80],[80,120],[50,120]]
        #lane_combinations = [[30, 60, 90, 120, 150]]
        lane_combinations = [[50, 80, 120]]
        number_cars = [10,15]
        car_speeds = [3]

        self.car_speed = random.choice(car_speeds)
        self.player_speed = 0
        #self.lanes = random.choices(lane_combinations, weights=lane_weights, k=1)[0]
        self.lanes = random.choice(lane_combinations)
        self.max_cars = random.choice(number_cars)
        
        lane_speeds = {30: self.car_speed - 2, 60: self.car_speed - 1, 90: self.car_speed, 120: self.car_speed - 1, 150: self.car_speed - 2}
        #lane_speeds = {50: self.car_speed - 2, 80: self.car_speed, 120: self.car_speed-1}
        self.cars = []
        lane_car_count = {lane: 0 for lane in self.lanes}

        for _ in range(self.max_cars):
            while True:
                lane = random.choice(self.lanes)
                if lane_car_count[lane] < 5:
                    car = {
                        'x': random.randint(0, self.window_width - self.car_width),
                        'lane': lane,
                        'speed': car_speeds[0]
                    }
                    self.cars.append(car)
                    lane_car_count[lane] += 1
                    break  # Exit the while loop once a valid lane is found

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
        reward = 0
        reward = -0.5
        self.episode_step += 1
        if self.episode_step % 1000 == 0:
            self.done = True
        previous_y = self.player_rect.y
        current_time = pygame.time.get_ticks()
        if action == 1 and self.episode_step>=30: # Up
            self.player_rect.y = max(0, self.player_rect.y - 5)
        elif action == 2 and self.episode_step>=30:  # Down
            self.player_rect.y = min(self.window_height - self.player_height, self.player_rect.y + 5)
        self.player_speed = self.player_rect.y - previous_y
        for car in self.cars:
            car['x'] += car['speed']
            if car['x'] > self.window_width:
                # car['x'] = -random.randint(100, 300)
                car['x'] = 0
                #car['speed'] = 
                #car['speed'] = self.car_speed 
                # print(f"Car speed: {car['speed']}")
                # print(f"Car lane: {car['lane']}")

        # Collision detection
        hit = any(self.player_rect.colliderect(pygame.Rect(car['x'], car['lane'], self.car_width, self.car_height)) for car in self.cars)
        if hit:
            self.player_rect.y = self.window_height - self.player_height - 10
            #reward -= 3
        
            self.last_time = current_time
        # if current_time - self.episode_start_time >= 60000:  # 60000 milliseconds = 1 minute
        #     self.done = True
            
        if self.player_rect.y <= 0:  # Reached top
            self.score +=1  
            reward += 10*(len(self.lanes))
            self.done = True

            

        
            self.player_rect.y = self.window_height - self.player_height - 10
            


        if self.observation_type == "pixel":
            self.update_frame_buffer()
            observation = self.get_observation()
        else:
            observation = self.get_object_data()

        return observation, reward, self.done, False, {}

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
            [self.player_rect.x, self.player_rect.y, self.player_speed, 0, 1, 0, 0],  # Player
            
        ] 
        # add lanes
        for lane in self.lanes:
            objects.append([self.window_width//2, lane, 0, 0, 0, 1, 0])

        for i, car in enumerate(self.cars):
            objects.append([car['x'], car['lane'], car['speed'], 0, 0, 0, 1])

        while len(objects) < 24:  # Ensure the list has a constant length
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



class FreewayEnvTest(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, render_mode='human', observation_type='pixel', frame_stack=4):
        super(FreewayEnvTest, self).__init__()
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

        self.lanes = [50, 80, 120]
        self.max_cars = 5
        self.action_space = spaces.Discrete(3)

        if observation_type == "pixel":
            self.observation_space = spaces.Box(low=0, high=255, shape=(self.frame_stack, 84, 84), dtype=np.uint8)
        else:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_cars + len(self.lanes) + 1, 7), dtype=np.float32)

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
                      'speed': random.randint(1, 2)} for _ in range(self.max_cars)]
        self.done = False
        self.episode_start_time = pygame.time.get_ticks()
        self.steps_since_collision = 0
        self.frame_buffer = np.zeros((self.frame_stack, 84, 84), dtype=np.uint8)
        if self.observation_type == "pixel":
            for _ in range(self.frame_stack):
                self.update_frame_buffer()
            return self.get_observation(), {}
        else:
            return self.get_object_data(), {}

    def step(self, action):
        reward = -0.5
        current_time = pygame.time.get_ticks()

        if self.steps_since_collision < 30:  # Check the step counter
            self.steps_since_collision += 1
        else:
            if action == 1:  # Up
                self.player_rect.y = max(0, self.player_rect.y - 5)
            elif action == 2:  # Down
                self.player_rect.y = min(self.window_height - self.player_height, self.player_rect.y + 5)

        for car in self.cars:
            car['x'] += car['speed']
            if car['x'] > self.window_width:
                car['x'] = 0
                car['speed'] = random.randint(1, 2)

        # Collision detection
        hit = any(self.player_rect.colliderect(pygame.Rect(car['x'], car['lane'], self.car_width, self.car_height)) for car in self.cars)
        if hit:
            self.player_rect.y = self.window_height - self.player_height - 10
            self.steps_since_collision = 0  # Reset the counter on collision

        if current_time - self.episode_start_time >= 60000:  # 60000 milliseconds = 1 minute
            self.done = True

        if self.player_rect.y <= 0:  # Reached top
            self.score += 1
            reward += 10 * len(self.lanes)
            self.player_rect.y = self.window_height - self.player_height - 10

        if self.observation_type == "pixel":
            self.update_frame_buffer()
            observation = self.get_observation()
        else:
            observation = self.get_object_data()

        return observation, reward, self.done, False, {}

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
        ]
        # add lanes
        for lane in self.lanes:
            objects.append([self.window_width // 2, lane, 0, 0, 0, 1, 0])

        for i, car in enumerate(self.cars):
            objects.append([car['x'], car['lane'], car['speed'], 0, 0, 0, 1])

        return torch.tensor(objects, dtype=torch.float32)

    def render(self, mode='human'):
        self.window.blit(self.background_image, (0, 0))
        for car in self.cars:
            self.window.blit(self.car_image, (car['x'], car['lane']))
        self.window.blit(self.player_image, (self.player_rect.x, self.player_rect.y))
        pygame.display.update()

    def close(self):
        pygame.quit()

import random
import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
from stable_baselines3 import PPO

class FreewayEnvDynamic(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, render_mode='human', observation_type='pixel', frame_stack=4):
        super(FreewayEnvDynamic, self).__init__()
        pygame.init()
        self.render_mode = render_mode
        self.observation_type = observation_type
        self.window_width = 210
        self.window_height = 160
        self.player_width = 5
        self.player_height = 5
        self.car_width = 20
        self.car_height = 20
        self.frame_stack = frame_stack
        self.max_episode_length = 1000
        self.steps_since_collision = 0
        self.collision_wait_steps = 30

        self.lanes = [50, 80, 120]
        self.max_cars = 5
        self.car_speed = 1
        self.action_space = spaces.Discrete(3)

        if observation_type == "pixel":
            self.observation_space = spaces.Box(low=0, high=255, shape=(self.frame_stack, 84, 84), dtype=np.uint8)
        else:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(24, 7), dtype=np.float32)

        self.window = pygame.display.set_mode((self.window_width, self.window_height))
        self.background_image = pygame.image.load("games/images/Atari - background.png")
        self.background_image = pygame.transform.scale(self.background_image, (self.window_width, self.window_height))
        self.player_image = pygame.image.load("games/images/chicken.png").convert_alpha()
        self.player_image = pygame.transform.scale(self.player_image, (self.player_width, self.player_height))
        self.car_image = pygame.image.load("games/images/car.png").convert_alpha()
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
        self.player_speed = 0
        self.score = 0
        self.done = False
        self.episode_step = 0
        self.episode_start_time = pygame.time.get_ticks()
        self.steps_since_collision = 0
        self.frame_buffer = np.zeros((self.frame_stack, 84, 84), dtype=np.uint8)
        self._update_environment()

        if self.observation_type == "pixel":
            for _ in range(self.frame_stack):
                self.update_frame_buffer()
            return self.get_observation(), {}
        else:
            return self.get_object_data(), {}

    def _update_environment(self):
        lane_combinations = [[50, 80, 120], [50, 80], [80, 120]]
        number_cars = [5, 10]
        car_speeds = [3, 4]

        self.car_speed = random.choice(car_speeds)
        self.lanes = random.choice(lane_combinations)
        self.max_cars = random.choice(number_cars)

        lane_speeds = {50: self.car_speed , 80: self.car_speed-1, 120: self.car_speed-2}
        self.cars = []
        lane_car_count = {lane: 0 for lane in self.lanes}

        for _ in range(self.max_cars):
            lane = random.choice(self.lanes)
            if lane_car_count[lane] < 5:
                car = {'x': random.randint(0, self.window_width - self.car_width), 'lane': lane, 'speed': lane_speeds[lane]}
                self.cars.append(car)
                lane_car_count[lane] += 1

    def step(self, action):
        reward = -0.5
        self.episode_step += 1
        player_pos = self.player_rect.y
        if self.steps_since_collision < self.collision_wait_steps:
            self.steps_since_collision += 1
        else:
            if action == 1:  # Up
                self.player_rect.y = max(0, self.player_rect.y - 5)
            elif action == 2:  # Down
                self.player_rect.y = min(self.window_height - self.player_height, self.player_rect.y + 5)
            self.player_speed = self.player_rect.y - player_pos
        for car in self.cars:
            car['x'] += car['speed']
            if car['x'] > self.window_width:
                car['x'] = 0
                # car['speed'] = random.randint(1, 2)

        hit = any(self.player_rect.colliderect(pygame.Rect(car['x'], car['lane'], self.car_width, self.car_height)) for car in self.cars)
        if hit:
            self.player_rect.y = self.window_height - self.player_height - 10
            self.steps_since_collision = 0  # Reset the counter on collision

        if self.player_rect.y <= 0:  # Reached top
            self.score += 1
            reward += 10 * len(self.lanes)
            self.player_rect.y = self.window_height - self.player_height - 10
            self._update_environment()  # Update the environment after reaching the top

        if self.episode_step >= self.max_episode_length:
            self.done = True

        if self.observation_type == "pixel":
            self.update_frame_buffer()
            observation = self.get_observation()
        else:
            observation = self.get_object_data()

        return observation, reward, self.done, False, {}

    def update_frame_buffer(self):
        frame = self.render_to_array()
        grayscale = np.dot(frame[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
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
            [self.player_rect.x, self.player_rect.y, self.player_speed , 0, 1, 0, 0],  # Player
        ]
        for lane in self.lanes:
            objects.append([self.window_width // 2, lane, 0, 0, 0, 1, 0])

        for i, car in enumerate(self.cars):
            objects.append([car['x'], car['lane'], car['speed'], 0, 0, 0, 1])

        while len(objects) < 24:  # Ensure the list has a constant length
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



# if __name__ == "__main__":
#     env = FreewayEnv(render_mode='human', observation_type='pixel')

#     model = PPO.load("ppo_freeway_pixel")

#     obs, _ = env.reset()
#     done = False
#     total_reward = 0
#     while not done:
#         action, _ = model.predict(obs)
#         obs, reward, done, _, _ = env.step(action)
#         total_reward += reward
#         pygame.time.delay(50)
#         env.render()

#     print(f"Total reward: {total_reward}")



if __name__=="__main__":
    env = FreewayEnvTest(render_mode='human', observation_type='graph')

    #model = PPO.load("ppo_freeway_pixel")
    model = PPO.load("logs/Freeway-GNN-training/best_model.zip")
    #model = PPO.load("ppo_custom_heterognn")
    obs,_ = env.reset()
    done = False
    total_reward = 0
    n_episodes = 10
    for _ in range(n_episodes):
        obs,_ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            # if action == 2 or action == 0:
            #     print(f"Player speed: {env.player_speed}")
            #action = env.action_space.sample()
            obs, reward, done, _,_ = env.step(action)
            total_reward += reward
            pygame.time.delay(50)
            env.render()

        print(f"Total reward: {total_reward}")
        total_reward = 0
    # while not done:
    #     action, _ = model.predict(obs)
    #     #action = env.action_space.sample()
    #     obs, reward, done, _,_ = env.step(action)
    #     total_reward += reward
    #     pygame.time.delay(50)
    #     env.render()

    # print(f"Total reward: {total_reward}")



    # # Evaluate the agent
    # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1, render=True)
    # print(f"Mean reward: {mean_reward} ± {std_reward}")

    
# if __name__=="__main__":
#     #env = FreewayEnv(render_mode='human', observation_type='graph')
#     env = FreewayEnv(render_mode='human', observation_type='pixel')
#     model = PPO.load("ppo_freeway_pixel")
#     #model = PPO.load("ppo_freeway_pixel_1.zip")
#     #model = PPO.load("logs/Freeway-GNN-training/best-1.zip")
#     #model = PPO.load("logs/Freeway-GNN-training/best_model.zip")
#     #model = PPO.load("logs/Freeway-GNN-training/best-threelanes-constant-speed.zip")
#     #model = PPO.load("logs/Freeway-GNN-eval/best_model.zip")
#     #model = PPO.load("freeway-gnn-eval.zip")

#     #model = PPO.load("freeway_GNN_training.zip")

#     #model = PPO.load("ppo_freeway_curriculum_1.zip")

#     #model = PPO.load("ppo_freeway_curriculum_2.zip")

#     # # Evaluate the agent
#     # mean_reward, std_reward = evaluate_policy(model, env, n_eval¸_episodes=1, render=True)
#     # print(f"Mean reward: {mean_reward} ± {std_reward}")

#     obs,_ = env.reset()
#     done = False
#     total_reward = 0
#     n_episodes = 10
#     for _ in range(n_episodes):
#         obs,_ = env.reset()
#         done = False
#         while not done:
#             action, _ = model.predict(obs)
#             if action == 2 or action == 0:
#                 print(f"Player speed: {env.player_speed}")
#             #action = env.action_space.sample()
#             obs, reward, done, _,_ = env.step(action)
#             total_reward += reward
#             pygame.time.delay(50)
#             env.render()

#         print(f"Total reward: {total_reward}")
#     # while not done:
#     #     action, _ = model.predict(obs)
#     #     #action = env.action_space.sample()
#     #     obs, reward, done, _,_ = env.step(action)
#     #     total_reward += reward
#     #     pygame.time.delay(50)
#     #     env.render()

#     # print(f"Total reward: {total_reward}")

