import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np 
import random
import torch
from torch_geometric.data import Data
import torch.nn.functional as F

class PongEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode='human', paddle_width=10, paddle_height=40, ball_size=15, paddle_speed=5):
        pygame.init()
        self.width = 800
        self.height = 600
        self.render_mode = render_mode
        self.paddle_width = paddle_width
        self.paddle_height = paddle_height
        self.ball_size = ball_size
        self.paddle_speed = paddle_speed
        self.action_space = spaces.Discrete(3)  # [Stay, Up, Down]
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.height, self.width), dtype=np.uint8)
        if self.render_mode == "human":
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Pong")
        else:
            self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock() 
        self.ai_reaction_time = 2  # milliseconds
        self.np_random = None
        self.frame_buffer = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        self.proximity_threshold = 50
    
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    
    def reset(self, seed = None, options = None):
        super().reset(seed = seed, options = options) 
        if seed is not None:
            self.seed(seed)  # Seed the RNG for the environment
        self.ball = pygame.Rect(self.width // 2 - self.ball_size // 2, self.height // 2 - self.ball_size // 2, self.ball_size, self.ball_size)
        self.left_paddle = pygame.Rect(50, self.height // 2 - self.paddle_height // 2, self.paddle_width, self.paddle_height)
        self.right_paddle = pygame.Rect(self.width - 50 - self.paddle_width, self.height // 2 - self.paddle_height // 2, self.paddle_width, self.paddle_height)
        self.ai_last_reaction_time = pygame.time.get_ticks()
        self.ball_speed_x, self.ball_speed_y = 4 * random.choice((1, -1)), 4 * random.choice((1, -1))
        self.left_player_score = 0
        self.right_player_score = 0
        return self._get_observation(), {}


    def render(self):
        if self.render_mode == "human":
            self.screen.fill((0, 0, 0))
            pygame.draw.rect(self.screen, (255, 255, 255), self.left_paddle)
            pygame.draw.rect(self.screen, (255, 255, 255), self.right_paddle)
            pygame.draw.ellipse(self.screen, (255, 255, 255), self.ball)
            pygame.display.flip()
            
            self.clock.tick(self.metadata['render_fps'])
        elif self.render_mode == "rgb_array":
            return np.array(pygame.surfarray.pixels3d(self.screen))

    def _get_observation(self):
        self.render()
        if self.render_mode == "rgb_array":
            frame = pygame.surfarray.array3d(pygame.display.get_surface())
            # Convert to grayscale
            grayscale = np.dot(frame[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
            # Update frame buffer, pushing back older frames
            self.frame_buffer = np.roll(self.frame_buffer, shift=-1, axis=2)
            self.frame_buffer[:, :, 3] = grayscale
            return self.frame_buffer
        else:
            # Return a dummy observation if not in rgb_array mode
            return self.frame_buffer
        return None
    
    def _apply_action(self, action):
        if action == 1 and self.left_paddle.top > 0:
            self.left_paddle.y -= self.paddle_speed
        elif action == 2 and self.left_paddle.bottom < self.height:
            self.left_paddle.y += self.paddle_speed


    def ai_move(self):
        current_time = pygame.time.get_ticks()
        # AI paddle movement
        if current_time - self.ai_last_reaction_time > self.ai_reaction_time:
            if self.ball.y < self.right_paddle.y + self.paddle_height / 2 and self.right_paddle.top > 0:
                self.right_paddle.y -= self.paddle_speed
            if self.ball.y > self.right_paddle.y + self.paddle_height / 2 and self.right_paddle.bottom < self.height:
                self.right_paddle.y += self.paddle_speed
            self.ai_last_reaction_time = current_time


    def _update_game_state(self):
        # Existing game state update logic
        self.ball.x += self.ball_speed_x
        self.ball.y += self.ball_speed_y
        # Check for collisions with top and bottom of the screen
        if self.ball.top <= 0 or self.ball.bottom >= self.height:
            self.ball_speed_y *= -1
        # AI paddle move
        self.ai_move()
        # Check for collisions with paddles
        collision = False
        if self.ball.colliderect(self.left_paddle) or self.ball.colliderect(self.right_paddle):
            self.ball_speed_x *= -1
            collision = True
            # Adjust the ball's position to prevent sticking
            if self.ball.colliderect(self.left_paddle):
                self.ball.left = self.left_paddle.right  # Place the ball right outside the left paddle
            elif self.ball.colliderect(self.right_paddle):
                self.ball.right = self.right_paddle.left  # Place the ball right outside the right paddle

        # Check for scoring
        score = 0
        if self.ball.left <= 0:
            self.right_player_score += 1
            score = -1  # Negative reward for the agent 
            self.ball_reset()
        elif self.ball.right >= self.width:
            self.left_player_score += 1
        
            score = 1   # Positive reward for the agent
            self.ball_reset()

        return collision, score
    

    def ball_reset(self):
        self.ball.x, self.ball.y = self.width // 2 - self.ball_size // 2, self.height // 2 - self.ball_size // 2
        self.ball_speed_x, self.ball_speed_y = 4 * random.choice((1, -1)), 4 * random.choice((1, -1))


    def step(self, action):
        self._apply_action(action)
        collision, score = self._update_game_state()
        observation = self._get_observation()
        reward = 0
        if collision:
            reward += 0.1  # Reward for hitting the ball
        reward += score  # Reward or penalty for scoring/losing a point
        done = self._check_done()
        return observation, reward, done, False, {}

    

    def _check_done(self):
        # Define the conditions under which the game is considered done
        if self.left_player_score >= 3 or self.right_player_score >= 3:
            return True
        return False

    def close(self):
        pygame.display.quit()
        pygame.quit()

class PongEnvRel(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode='human', paddle_width=10, paddle_height=40, ball_size=15, paddle_speed=5):
        pygame.init()
        self.width = 800
        self.height = 600
        self.render_mode = render_mode
        self.paddle_width = paddle_width
        self.paddle_height = paddle_height
        self.ball_size = ball_size
        self.paddle_speed = paddle_speed
        self.action_space = spaces.Discrete(3)  # [Stay, Up, Down]
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.height, self.width), dtype=np.uint8)
        if self.render_mode == "human":
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Pong")
        else:
            self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock() 
        self.ai_reaction_time = 2  # milliseconds
        self.np_random = None
        self.frame_buffer = np.zeros((self.height, self.width, 4), dtype=np.uint8)
    
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    
    def reset(self, seed = None, options = None):
        super().reset(seed = seed, options = options) 
        if seed is not None:
            self.seed(seed)  # Seed the RNG for the environment
        self.ball = pygame.Rect(self.width // 2 - self.ball_size // 2, self.height // 2 - self.ball_size // 2, self.ball_size, self.ball_size)
        self.left_paddle = pygame.Rect(50, self.height // 2 - self.paddle_height // 2, self.paddle_width, self.paddle_height)
        self.right_paddle = pygame.Rect(self.width - 50 - self.paddle_width, self.height // 2 - self.paddle_height // 2, self.paddle_width, self.paddle_height)
        self.ai_last_reaction_time = pygame.time.get_ticks()
        self.ball_speed_x, self.ball_speed_y = 4 * random.choice((1, -1)), 4 * random.choice((1, -1))
        self.left_player_score = 0
        self.right_player_score = 0
        return self._get_observation(), {}


    def render(self):
        if self.render_mode == "human":
            self.screen.fill((0, 0, 0))
            pygame.draw.rect(self.screen, (255, 255, 255), self.left_paddle)
            pygame.draw.rect(self.screen, (255, 255, 255), self.right_paddle)
            pygame.draw.ellipse(self.screen, (255, 255, 255), self.ball)
            pygame.display.flip()
            
            self.clock.tick(self.metadata['render_fps'])
        elif self.render_mode == "rgb_array":
            return np.array(pygame.surfarray.pixels3d(self.screen))

    def _get_observation(self):
        self.render()
        if self.render_mode == "rgb_array":
            frame = pygame.surfarray.array3d(pygame.display.get_surface())
            # Convert to grayscale
            grayscale = np.dot(frame[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
            # Update frame buffer, pushing back older frames
            self.frame_buffer = np.roll(self.frame_buffer, shift=-1, axis=2)
            self.frame_buffer[:, :, 3] = grayscale
            return self.frame_buffer
        else:
            # Return a dummy observation if not in rgb_array mode
            return self.frame_buffer
        return None
    
    def _apply_action(self, action):
        if action == 1 and self.left_paddle.top > 0:
            self.left_paddle.y -= self.paddle_speed
        elif action == 2 and self.left_paddle.bottom < self.height:
            self.left_paddle.y += self.paddle_speed


    def ai_move(self):
        current_time = pygame.time.get_ticks()
        # AI paddle movement
        if current_time - self.ai_last_reaction_time > self.ai_reaction_time:
            if self.ball.y < self.right_paddle.y + self.paddle_height / 2 and self.right_paddle.top > 0:
                self.right_paddle.y -= self.paddle_speed
            if self.ball.y > self.right_paddle.y + self.paddle_height / 2 and self.right_paddle.bottom < self.height:
                self.right_paddle.y += self.paddle_speed
            self.ai_last_reaction_time = current_time


    def _update_game_state(self):
        # Existing game state update logic
        self.ball.x += self.ball_speed_x
        self.ball.y += self.ball_speed_y
        # Check for collisions with top and bottom of the screen
        if self.ball.top <= 0 or self.ball.bottom >= self.height:
            self.ball_speed_y *= -1
        # AI paddle move
        self.ai_move()
        # Check for collisions with paddles
        collision = False
        if self.ball.colliderect(self.left_paddle) or self.ball.colliderect(self.right_paddle):
            self.ball_speed_x *= -1
            collision = True
            # Adjust the ball's position to prevent sticking
            if self.ball.colliderect(self.left_paddle):
                self.ball.left = self.left_paddle.right  # Place the ball right outside the left paddle
            elif self.ball.colliderect(self.right_paddle):
                self.ball.right = self.right_paddle.left  # Place the ball right outside the right paddle

        # Check for scoring
        score = 0
        if self.ball.left <= 0:
            self.right_player_score += 1
            score = -1  # Negative reward for the agent 
            self.ball_reset()
        elif self.ball.right >= self.width:
            self.left_player_score += 1
        
            score = 1   # Positive reward for the agent
            self.ball_reset()

        return collision, score
    

    def ball_reset(self):
        self.ball.x, self.ball.y = self.width // 2 - self.ball_size // 2, self.height // 2 - self.ball_size // 2
        self.ball_speed_x, self.ball_speed_y = 4 * random.choice((1, -1)), 4 * random.choice((1, -1))


    def step(self, action):
        self._apply_action(action)
        collision, score = self._update_game_state()
        observation = self._get_observation()
        reward = 0
        if collision:
            reward += 0.1  # Reward for hitting the ball
        reward += score  # Reward or penalty for scoring/losing a point
        done = self._check_done()
        graph_data = self.get_graph_data()
        return observation, reward, done, False, {}

    
    def get_graph_data(self):
        # Feature vectors setup
        ball_features = [self.ball.x, self.ball.y, self.ball_speed_x, self.ball_speed_y, 1, 0, 0, 0, 0]
        left_paddle_features = [self.left_paddle.x,self.left_paddle.y , 0, 0, 0, 1, 0, 0, 0, 0]
        right_paddle_features = [self.right_paddle.x, self.right_paddle.y, 0, 0, 0, 0, 1, 0, 0]
        top_wall_features = [0, self.height, 0, 0, 0, 0, 0, 1, 0]
        # Existing code
        bottom_wall_features = [0, 0, 0, 0, 0, 0, 0, 0, 1]

        # Combine all features into a single tensor
        features = torch.tensor([ball_features, left_paddle_features, right_paddle_features, top_wall_features, bottom_wall_features], dtype=torch.float)

        # Calculate edges based on iscloseTo predicate
        objects = [(self.ball.x, self.ball.y), (self.left_paddle.x, self.left_paddle.y), (self.right_paddle.x, self.right_paddle.y), (0, self.height), (0, 0)]
        edge_index = []
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                dist = np.linalg.norm(np.array(objects[i]) - np.array(objects[j]))
                if dist < self.proximity_threshold:
                    edge_index.append([i, j])
                    edge_index.append([j, i])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        # Feature vector for each edge
        edge_features = [1.0] * edge_index.size(1)
        edge_attr = torch.tensor(edge_features, dtype=torch.float).view(-1, 1)

        # Create the PyTorch Geometric Data object
        data = Data(x=features, edge_index=edge_index, edge_attr=edge_attr)

        # Create the PyTorch Geometric Data object
        return data



    def _check_done(self):
        # Define the conditions under which the game is considered done
        if self.left_player_score >= 3 or self.right_player_score >= 3:
            return True
        return False

    def close(self):
        pygame.display.quit()
        pygame.quit()


if __name__ == "__main__":
    env = PongEnv(render_mode='human')
    env.reset()

    done = False
    try:
        while not done:
            action = env.action_space.sample()
            _, _, done, _, _ = env.step(action)
            env.render()
            pygame.time.wait(10)
    finally:
        print(done)
        env.close() 
