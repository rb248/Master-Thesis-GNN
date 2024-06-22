import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import random
import torch
from torch_geometric.data import Data
import stable_baselines3.common.env_checker
from skimage.transform import resize

class PongEnvNew(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60, "observation_types": ["pixel", "graph"]}

    def __init__(self, render_mode='human', observation_type='pixel', paddle_width=5, paddle_height=40, ball_size=5, paddle_speed=5, frame_stack=4):
        pygame.init()
        self.width = 700
        self.height = 500
        self.render_mode = render_mode
        self.observation_type = observation_type
        self.paddle_width = paddle_width
        self.paddle_height = paddle_height
        self.ball_size = ball_size
        self.paddle_speed = paddle_speed
        self.frame_stack = frame_stack  # Number of frames to stack
        self.action_space = spaces.Discrete(3)  # [Stay, Up, Down]

        if observation_type == "pixel":
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.frame_stack, 84, 84), dtype=np.uint8)
        else:
            # Define a generic observation space for graph data
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3, 7), dtype=np.float32)  # Number of objects and feature length

        if self.render_mode == "human" or self.render_mode == "rgb_array":
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Pong")
        else:
            self.screen = pygame.Surface((self.width, self.height))
        
        self.clock = pygame.time.Clock()
        self.ai_reaction_time = 10  # milliseconds
        self.np_random = None
        self.frame_buffer = np.zeros((self.height, self.width, self.frame_stack), dtype=np.uint8)
        self.proximity_threshold = 50
    
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        if not pygame.display.get_init():
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
        if seed is not None:
            self.seed(seed)  # Seed the RNG for the environment
        self.ball = pygame.Rect(self.width // 2 - self.ball_size // 2, self.height // 2 - self.ball_size // 2, self.ball_size, self.ball_size)
        self.left_paddle = pygame.Rect(20, self.height // 2 - self.paddle_height // 2, self.paddle_width, self.paddle_height)
        self.right_paddle = pygame.Rect(self.width - 20 - self.paddle_width, self.height // 2 - self.paddle_height // 2, self.paddle_width, self.paddle_height)
        self.ai_last_reaction_time = pygame.time.get_ticks()
        self.ball_speed_x, self.ball_speed_y = 2 * random.choice((1, -1)), 2 * random.choice((1, -1))
        self.left_player_score = 0
        self.right_player_score = 0
        self.frame_buffer = np.zeros((self.height, self.width, self.frame_stack), dtype=np.uint8)
        
        # Fill the frame buffer with the initial frame
        if self.observation_type == "pixel":
            for _ in range(self.frame_stack):
                self._get_observation()
        else:
            return self.get_graph_data(), {}

        return self._get_observation(), {}

    def render(self):
        if not self.render_mode in ['human', 'rgb_array']:
            # If not in a mode that requires rendering, skip the rendering.
            return None
        if not pygame.display.get_init():
            # Reinitialize display if needed
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
        try:
            self.screen.fill((0, 0, 0))
            pygame.draw.rect(self.screen, (255, 255, 255), self.left_paddle)
            pygame.draw.rect(self.screen, (255, 255, 255), self.right_paddle)
            pygame.draw.ellipse(self.screen, (255, 255, 255), self.ball)

            if self.render_mode == "human":
                pygame.display.flip()
            elif self.render_mode == "rgb_array":
                return np.array(pygame.surfarray.pixels3d(self.screen))
        except pygame.error as e:
            print("Pygame error:", e)
            return None

    def _get_observation(self):
        if self.observation_type == "pixel":
            self.render()
            frame = pygame.surfarray.array3d(pygame.display.get_surface())
            frame = np.transpose(frame, (1, 0, 2))  # Transpose to match (height, width, channels)
            grayscale = np.dot(frame[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)  # Convert to grayscale
            self.frame_buffer = np.roll(self.frame_buffer, shift=-1, axis=2)
            self.frame_buffer[:, :, -1] = grayscale  # Update the last frame

            # Resize the frame buffer to 84x84 pixels
            resized_frame_buffer = np.zeros((84, 84, self.frame_stack), dtype=np.uint8)
            for i in range(self.frame_stack):
                resized_frame_buffer[:, :, i] = resize(self.frame_buffer[:, :, i], (84, 84), anti_aliasing=True, preserve_range=True).astype(np.uint8)

            # Convert to (batch_size, 4, 84, 84)
            return resized_frame_buffer.transpose(2, 0, 1)
        else:
            return self.get_graph_data()

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
        # Update the ball's position
        self.ball.x += self.ball_speed_x
        self.ball.y += self.ball_speed_y

        # Debugging statements to track ball position and speed

        # Check for collisions with the top and bottom of the screen
        if self.ball.top <= 0:
            self.ball.top = 1  # Prevent sticking to the top wall
            self.ball_speed_y *= -1
        elif self.ball.bottom >= self.height:
            self.ball.bottom = self.height - 1  # Prevent sticking to the bottom wall
            self.ball_speed_y *= -1

        # AI paddle move
        self.ai_move()

        # Initialize collision and score
        collision = False
        score = 0

        # Ball collision check on gutters or paddles
        if (self.ball.left <= self.left_paddle.right and 
            self.left_paddle.top < self.ball.centery < self.left_paddle.bottom):
            collision = True
            self.ball_speed_x = -self.ball_speed_x * 1.1
            self.ball_speed_y *= 1.1
            # Adjust the ball's position to prevent sticking
            self.ball.left = self.left_paddle.right
        elif self.ball.left <= 0:
            self.right_player_score += 1
            score = -1  # Negative reward for the agent
            self.ball_reset()
            return collision, score  # Early return to avoid further processing
        
        if (self.ball.right >= self.right_paddle.left and 
            self.right_paddle.top < self.ball.centery < self.right_paddle.bottom):
            collision = True
            self.ball_speed_x = -self.ball_speed_x * 1.1
            self.ball_speed_y *= 1.1
            # Adjust the ball's position to prevent sticking
            self.ball.right = self.right_paddle.left
        elif self.ball.right >= self.width:
            self.left_player_score += 1
            score = 1  # Positive reward for the agent
            self.ball_reset()
            return collision, score  # Early return to avoid further processing

        return collision, score


    def ball_reset(self):
        self.ball.x = self.width // 2 - self.ball_size // 2
        self.ball.y = self.height // 2 - self.ball_size // 2
        self.ball_speed_x = 2 * random.choice((1, -1))
        self.ball_speed_y = 2 * random.choice((1, -1))
        # Debugging statement to confirm ball reset

    def step(self, action):
        if not pygame.display.get_init():
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.display.quit()
                pygame.quit()
                raise SystemExit("Pygame QUIT event received.")
        self._apply_action(action)
        collision, score = self._update_game_state()
        observation = self._get_observation()
        reward = 0
        if collision:
            reward += 0.1  # Reward for hitting the ball
        reward += score  # Reward for scoring
    
        # Additional shaping rewards (optional)
        if not collision and self.ball_speed_x > 0:  # Encourages the agent to stay in advantageous positions
            reward += 0.01  # Small reward for keeping the ball in play
        done = self._check_done()
        if done:
            if self.left_player_score >= 20:
                reward = 1
            elif self.right_player_score >= 20:
                reward = -1
        
        info = {}
        truncated = False
        return observation, reward, done, truncated, info

    def get_graph_data(self):
        # Define the features for each object in the environment
        objects = {
            "ball": [self.ball.x, self.ball.y, self.ball_speed_x, self.ball_speed_y, 1, 0, 0],
            "left_paddle": [self.left_paddle.x, self.left_paddle.y, 0, 0, 0, 1, 0],
            "right_paddle": [self.right_paddle.x, self.right_paddle.y, 0, 0, 0, 1, 0],
            # "top_wall": [0, 0, 0, 0, 0, 0, 1],
            # "bottom_wall": [0, self.height, 0, 0, 0, 0, 1]
        }

        # Convert the object features to a tensor
        node_features = [features for features in objects.values()]
        x = torch.tensor(node_features, dtype=torch.float32)

        return x 

    def _check_done(self):
        # Define the conditions under which the game is considered done
        if self.left_player_score >= 20 or self.right_player_score >= 20:
            return True
        return False

    def close(self):
        pygame.display.quit()
        pygame.quit()

from stable_baselines3 import PPO

if __name__ == "__main__":
    env = PongEnvNew(render_mode='human', observation_type='graph')
    model = PPO.load("ppo_custom_heterognn_new")
    num_episodes = 100

    for i_episode in range(num_episodes):
        done = False
        obs, _ = env.reset()  # Reset the environment at the start of each episode
        try:
            while not done:
                action,_ = model.predict(obs)
                obs, _, done, _, _ = env.step(action)
                env.render()
                pygame.time.wait(10)
        except Exception as e:
            print(f"An error occurred: {e}")
            break
        finally:
            env.close()
