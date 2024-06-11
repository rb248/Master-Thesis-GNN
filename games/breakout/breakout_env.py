import gym
from gym import spaces
import numpy as np
import pygame
from games.breakout.paddle import Paddle
from games.breakout.ball import Ball
from games.breakout.scoreboard import Scoreboard
from games.breakout.ui import UI
from games.breakout.bricks import Bricks
import time
import torch
from torch_geometric.data import Data, HeteroData 
import networkx as nx
from games.encoder.GraphEncoder import GraphConverter

class BreakoutEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, num_frames=4):
        super(BreakoutEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.num_frames = num_frames
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(600, 1200, 3 * num_frames),  # Stack frames along the channel dimension
                                            dtype=np.uint8)
        self.proximity_threshold = 50  # Example threshold for proximity
        self.action_space = spaces.Discrete(3)  # actions: move left, stay, move right
        # Example for observation space: the game state
        self.observation_space = spaces.Box(low=0, high=255, shape=(600, 1200, 3), dtype=np.uint8)
        self.screen_width = 1200
        self.screen_height = 600
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption('Breakout') 

        pygame.init()
        self.clock = pygame.time.Clock()
        self.frame_buffer = np.zeros((self.screen_height, self.screen_width, 3 * num_frames), dtype=np.uint8)

        

    def reset(self):
        self.paddle = Paddle(self.screen)
        self.ball = Ball(self.screen)
        self.bricks = Bricks(self.screen)
        self.scoreboard = Scoreboard(self.screen, lives=5 )
        self.ui = UI(self.screen)
        self.paddle.draw()
        self.ball.draw()
        self.bricks.draw()
        self.scoreboard.draw()
        self.ui.header()
        initial_state = self.get_state()
        for i in range(self.num_frames):
            start_idx = i * 3
            self.frame_buffer[:, :, start_idx:start_idx + 3] = initial_state
        return self.frame_buffer


    def check_collision_with_walls(self, ball, score, ui):
        reward = 0
        # detect collision with left and right walls:
        if ball.rect.left <= 0 or ball.rect.right >= self.screen_width:
            ball.bounce(x_bounce=True, y_bounce=False)

        # detect collision with upper wall
        if ball.rect.top <= 0:
            ball.bounce(x_bounce=False, y_bounce=True)

        # detect collision with bottom wall
        if ball.rect.bottom >= self.screen_height:
            ball.reset()
            reward = -100
            score.decrease_lives()
            
            if score.lives == 0:
                score.reset()
                playing_game = False
                ui.game_over(win=False)
            else:
                ui.change_color() 
        return reward


    def check_collision_with_paddle(self,ball, paddle):
        reward = 0
        if ball.rect.colliderect(paddle.rect):
            # Determine the collision side and bounce accordingly
            center_ball = ball.rect.centerx
            center_paddle = paddle.rect.centerx
            reward = 10

            if center_ball < center_paddle:  # Ball hits the left side of the paddle
                ball.bounce(x_bounce=True, y_bounce=True)
            elif center_ball > center_paddle:  # Ball hits the right side of the paddle
                ball.bounce(x_bounce=True, y_bounce=True)
            else:
                # Ball hits the middle of the paddle
                ball.bounce(x_bounce=False, y_bounce=True)
        return reward


    def check_collision_with_bricks(self, ball, score, bricks):
        reward = 0
        for brick in bricks.bricks:
            if ball.rect.colliderect(brick.rect):
                score.increase_score()
                reward += 100
                brick.quantity -= 1
                if brick.quantity == 0:
                    bricks.bricks.remove(brick)
                # Determine collision direction
                # Note: Simple version without precise side detection
                ball.bounce(x_bounce=False, y_bounce=True)
                break 
        return reward
        

        

    def step(self, action):
        assert self.action_space.contains(action), f"{action} is an invalid action"
        
        # Clear the screen (fill with black or another color)
        self.screen.fill((0, 0, 0))
        
        # Map action to game movements
        if action == 0:
            self.paddle.move_left()
        elif action == 1:
            self.paddle.move_right()
        elif action == 2:
            pass  # Do nothing for 'stay' action

        # Move ball and check for interactions
        self.ball.move()
        
        # Check collisions and compute rewards
        reward = 0
        reward += self.check_collision_with_walls(self.ball, self.scoreboard, self.ui)
        reward += self.check_collision_with_paddle(self.ball, self.paddle)
        reward += self.check_collision_with_bricks(self.ball, self.scoreboard, self.bricks)
        new_frame = self.get_state()
        # Update the frame buffer
        self.update_frame_buffer(new_frame)
        # Draw all game elements
        self.paddle.draw()
        self.ball.draw()
        self.bricks.draw()
        self.scoreboard.draw()
        self.get_graph_data()
        # Check if game over
        done = self.scoreboard.lives == 0 or len(self.bricks.bricks) == 0

        # Update the display to show the new positions of game elements
        pygame.display.flip()

        # Additional info can be passed, though not used here
        info = {}

        return self.frame_buffer, reward, done, info
    

   


    def check_proximity(self, rect1, rect2, d=50):
        # Dummy implementation for proximity check
        return np.linalg.norm(np.array([rect1['x'], rect1['y']]) - np.array([rect2['x'], rect2['y']])) < d

    def check_adjacent(self, rect1, rect2, d=50):
        # Dummy implementation for adjacency check
        return np.linalg.norm(np.array([rect1['x'], rect1['y']]) - np.array([rect2['x'], rect2['y']])) < d

    def get_graph_data(self):
        # Initialize a NetworkX graph
        graph = nx.Graph()

        # Define object features and add nodes
        ball_features = [self.ball.rect.x, self.ball.rect.y, self.ball.x_move_dist, self.ball.y_move_dist, 1, 0, 0]
        graph.add_node("ball", type="object", features=ball_features)

        paddle_features = [self.paddle.rect.x, self.paddle.rect.y, 0, 0, 0, 1, 0]
        graph.add_node("paddle", type="object", features=paddle_features)

        brick_features = [[brick.rect.x, brick.rect.y, 0, 0, 0, 0, 1] for brick in self.bricks.bricks]
        for i, features in enumerate(brick_features):
            graph.add_node(f"brick_{i}", type="object", features=features)

        # Combine object positions
        object_positions = {
            "ball": ball_features[:2],
            "paddle": paddle_features[:2],
        }
        for i, features in enumerate(brick_features):
            object_positions[f"brick_{i}"] = features[:2]
        # Proximity threshold for creating atoms
        proximity_threshold = self.proximity_threshold

        # Create atom nodes and edges based on proximity and adjacency
        atom_index = len(object_positions)  # Start indexing atoms after all objects
        standard_feature_vector_size = len(ball_features)
        empty_feature_vector = [0] * (2 * standard_feature_vector_size)


        # Add proximity atoms and edges for ball and bricks
        for i, brick in enumerate(self.bricks.bricks):
            if self.check_proximity(self.ball, brick, d=50):
                atom_node = f"Proximity_Ball_Brick_{i}_{atom_index}"
                graph.add_node(atom_node, type="atom", features=empty_feature_vector, predicate="Proximity")
                graph.add_edge("ball", atom_node, position=0)
                graph.add_edge(f"brick_{i}", atom_node, position=1)
                atom_index += 1
        # Add proximity atoms and edges for paddle and ball
        if self.check_proximity(self.ball, self.paddle, d=50):
            atom_node = f"Proximity_Ball_Paddle_{atom_index}"
            graph.add_node(atom_node, type="atom",features=empty_feature_vector, predicate="Proximity")
            graph.add_edge("ball", atom_node, position=0)
            graph.add_edge("paddle", atom_node, position=1)
            atom_index += 1

        # Add adjacent atoms and edges (bricks with bricks)
        for i, brick1 in enumerate(self.bricks.bricks):
            for j, brick2 in enumerate(self.bricks.bricks):
                if i != j and self.check_adjacent(brick1, brick2, d=5):
                    atom_node = f"Adjacent_Bricks_{i}_{j}_{atom_index}"
                    graph.add_node(atom_node, type="atom", features=empty_feature_vector,predicate="Adjacent")
                    graph.add_edge(f"brick_{i}", atom_node, position=0)
                    graph.add_edge(f"brick_{j}", atom_node, position=1)
                    atom_index += 1

        # Create a GraphConverter object
        converter = GraphConverter()

        # Convert the NetworkX graph to a PyG Data object
        data = converter.to_pyg_data(graph)
        return data

    def check_collision(self, rect1, rect2):
        return rect1.colliderect(rect2)
    
    def check_proximity(self, obj1, obj2, d):
        center1 = obj1.rect.center
        center2 = obj2.rect.center
        distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        return distance < d
    
    def check_directional_influence(self, rect1, rect2, theta):
        # Check if rect1 is moving towards rect2 within a given angle theta
        # Simplified for demonstration
        return True  # Add your own logic here
    
    def check_adjacent(self, rect1, rect2, d):
        return self.check_proximity(rect1, rect2, d)


    def render(self, mode='human'):

        pass # update turtle graphics if needed

    def close(self):
        pygame.quit()

    def check_collisions(self):
        # Implement collision checks
        reward = 0
        # Implement collision logic with walls, paddle, bricks
        # Adjust reward accordingly
        return reward

    def get_state(self):
        # Get the surface array from Pygame
        surface_array = pygame.surfarray.array3d(pygame.display.get_surface())
        # Transpose the array from (width, height, channels) to (height, width, channels)
        transposed_array = np.transpose(surface_array, axes=(1, 0, 2))
        return transposed_array 
    
    def update_frame_buffer(self, new_frame):
        # Shift frames to the left in the buffer and append the new frame on the right
        # Ensure that the new_frame is transposed before being added to the frame buffer
        self.frame_buffer = np.roll(self.frame_buffer, shift=-3, axis=2)
        self.frame_buffer[:, :, -3:] = new_frame
    
if __name__ == "__main__":
    env = BreakoutEnv()
    env.reset()
    for _ in range(1000):
        env.step(env.action_space.sample())
        time.sleep(0.1)
    env.close()
