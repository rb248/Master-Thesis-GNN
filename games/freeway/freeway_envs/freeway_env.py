import random
import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import torch_geometric
from torch_geometric.data import Data
from ...encoder.GraphEncoder import GraphConverter
import networkx as nx

class FreewayEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self):
        super(FreewayEnv, self).__init__()
        pygame.init()
        self.window_width = 800
        self.window_height = 600
        self.player_width = 30
        self.player_height = 30
        self.car_width = 50
        self.car_height = 50 

        self.lanes = [100, 200, 300, 400, 500, 600, 700]

        # Define action and observation space
        # Actions: 0 - Stay, 1 - Move Up, 2 - Move Down
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(self.window_height, self.window_width, 3 * self.frame_stack),
                                            dtype=np.uint8)
        # Load images
        self.window = pygame.display.set_mode((self.window_width, self.window_height))

        self.background_image = pygame.transform.scale(pygame.image.load("games/images/Atari - background.png"), (self.window_width, self.window_height))
        self.player_image = pygame.transform.scale(pygame.image.load("games/images/chicken.png").convert_alpha(), (self.player_width, self.player_height))
        self.car_image = pygame.transform.scale(pygame.image.load("games/images/car2.png").convert_alpha(), (self.car_width, self.car_height))
        self.frame_buffer = np.zeros((self.window_height, self.window_width, 3 * self.frame_stack), dtype=np.uint8)

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
        self.frame_buffer = np.zeros((self.window_height, self.window_width, 3 * self.frame_stack), dtype=np.uint8)
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
        graph_data = self.get_graph_data()

        return self.get_observation(), self.score, self.done, {}
            
    def get_graph_data(self):
        # Initialize a NetworkX graph
        graph = nx.Graph()

        # Define object features and add nodes
        chicken_features = [self.player_rect.x, self.player_rect.y, 5, 1, 0, 0]
        graph.add_node("chicken", type="object", features=chicken_features)
        
        lane_features = [[0, lane, 0, 0, 1, 0] for lane in self.lanes]
        for i, features in enumerate(lane_features):
            graph.add_node(f"lane_{i}", type="object", features=features)

        car_features = [[car['x'], car['lane'], car['speed'], 0, 0, 1] for car in self.cars]
        for i, features in enumerate(car_features):
            graph.add_node(f"car_{i}", type="object", features=features)

        # Combine object positions
        object_positions = {
            "chicken": chicken_features[:2],
        }
        for i, lane in enumerate(self.lanes):
            object_positions[f"lane_{i}"] = lane_features[i][:2]
        for i, car in enumerate(self.cars):
            object_positions[f"car_{i}"] = car_features[i][:2]


        # Create atom nodes and edges based on proximity
        atom_index = len(object_positions)  # Start indexing atoms after all objects
        standard_feature_vector_size = len(chicken_features)
        empty_feature_vector = [0] *(2* standard_feature_vector_size)

        # Add ChickenOnLane atoms and edges
        for i, lane in enumerate(self.lanes):
            # check if the chicken is in the range of the lane of +-50
                if self.player_rect.y >= lane - 50 and self.player_rect.y <= lane + 50:
                    atom_node = f"ChickenOnLane_{atom_index}"
                    graph.add_node(atom_node, type="atom", features=empty_feature_vector, predicate="ChickenOnLane")
                    graph.add_edge("chicken", atom_node, position=0)
                    graph.add_edge(f"lane_{i}", atom_node, position=1)
                    atom_index += 1

        # Add CarOnLane atoms and edges
        num_lanes = len(self.lanes)
        for i, car in enumerate(self.cars, start=num_lanes + 1):
            car_lane_index = self.lanes.index(car['lane'])
            atom_node = f"CarOnLane_{atom_index}"
            graph.add_node(atom_node, type="atom",features=empty_feature_vector, predicate="CarOnLane")
            graph.add_edge(f"car_{i - num_lanes - 1}", atom_node, position=0)
            graph.add_edge(f"lane_{car_lane_index}", atom_node, position=1)
            atom_index += 1

        # Add LaneNextToLane atoms and edges
        for i in range(num_lanes - 1):
            atom_node = f"LaneNextToLane_{atom_index}"
            graph.add_node(atom_node, type="atom",features=empty_feature_vector, predicate="LaneNextToLane")
            graph.add_edge(f"lane_{i}", atom_node, position=0)
            graph.add_edge(f"lane_{i + 1}", atom_node, position=1)
            atom_index += 1

        # Create a GraphConverter object
        converter = GraphConverter()

        # Convert the NetworkX graph to a PyG Data object
        data = converter.to_pyg_data(graph)
        return data
    
    def render(self, mode='human'):
        self.window.blit(self.background_image, (0, 0))
        for car in self.cars:
            self.window.blit(self.car_image, (car['x'], car['lane']))
        self.window.blit(self.player_image, (self.player_rect.x, self.player_rect.y))
        pygame.display.update()

    def get_observation(self):
        # You can also choose to return other representations of the game state
        frame = pygame.surfarray.array3d(pygame.display.get_surface())
        frame = frame.transpose((1, 0, 2))  # Correct the shape to (height, width, channels)
        # Update frame buffer
        self.frame_buffer = np.roll(self.frame_buffer, -3, axis=2)
        self.frame_buffer[:, :, -3:] = frame
        return self.frame_buffer

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
