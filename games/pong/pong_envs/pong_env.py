import gymnasium as gym
from gymnasium import spaces
from matplotlib import pyplot as plt
import pygame
import numpy as np
import random
from skimage.color import rgb2gray
from skimage.transform import resize
from collections import deque
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

class PongEnvNew(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, observation_type='pixel', paddle_width=5, paddle_height=20, ball_size=5, paddle_speed=10, ai_paddle_speed=10, frame_stack=4, ai_reaction_delay=5, ball_speed = 7):
        super(PongEnvNew, self).__init__()
        self.width = 210
        self.height = 160
        self.render_mode = render_mode
        self.observation_type = observation_type
        self.paddle_width = paddle_width
        self.paddle_height = paddle_height
        self.ball_size = ball_size
        self.paddle_speed = paddle_speed
        self.ai_paddle_speed = ai_paddle_speed
        self.frame_stack = frame_stack
        self.ai_reaction_delay = ai_reaction_delay  # Number of frames to delay AI reaction
        self.ai_delay_counter = 0  # Initialize delay counter
        self.action_space = spaces.Discrete(3)  # [Stay, Up, Down]

        if observation_type == "pixel":
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=(frame_stack, 84, 84), dtype=np.uint8)
        else:
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3, 7), dtype=np.float32)

        self.screen = None
        self.offscreen_surface = None
        self.clock = None
        self.frame_buffer = deque(maxlen=frame_stack)
        self.ball_speed = ball_speed
    
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        if seed is not None:
            self.seed(seed)
        self.ball = pygame.Rect(self.width // 2 - self.ball_size // 2, self.height // 2 - self.ball_size // 2, self.ball_size, self.ball_size)
        self.left_paddle = pygame.Rect(20, self.height // 2 - self.paddle_height // 2, self.paddle_width, self.paddle_height)
        self.right_paddle = pygame.Rect(self.width - 20 - self.paddle_width, self.height // 2 - self.paddle_height // 2, self.paddle_width, self.paddle_height)
        
        angle = random.uniform(-np.pi / 4, np.pi / 4)
        
        self.ball_speed_x = self.ball_speed * np.cos(angle) * random.choice([1, -1])
        self.ball_speed_y = self.ball_speed * np.sin(angle) * random.choice([1, -1])
        
        self.left_player_score = 0
        self.right_player_score = 0
        self.frame_buffer.clear()
        self.ai_delay_counter = 0  # Reset the AI delay counter
        if self.observation_type == "pixel":
            obs = self._get_obs()
            for _ in range(self.frame_stack):
                self.frame_buffer.append(obs)
            return np.array(self.frame_buffer),{}
        else:
            return self.get_graph_data(),{}

    def step(self, action):
        self.ai_move()
        self._apply_action(action)
        reward, done = self._update_game_state()

        
        info = {}
        truncated = False
        if self.observation_type == "pixel":
            obs = self._get_obs()
            self.frame_buffer.append(obs)
            return np.array(self.frame_buffer), reward, done, truncated, info
        else:
            return self.get_graph_data(), reward, done, truncated, info

    def render(self, mode="human"):
        if mode == "human":
            if self.screen is None:
                pygame.init()
                self.screen = pygame.display.set_mode((self.width, self.height))
                pygame.display.set_caption("Pong")
                self.clock = pygame.time.Clock()
            self._render_on_surface(self.screen)
            pygame.display.flip()
            self.clock.tick(60)

    def close(self):
        pygame.quit()

    def _get_obs(self):
        if self.offscreen_surface is None:
            self.offscreen_surface = pygame.Surface((self.width, self.height))
        self._render_on_surface(self.offscreen_surface)
        frame = pygame.surfarray.array3d(self.offscreen_surface)
        frame = np.transpose(frame, (1, 0, 2))  # Transpose to match (height, width, channels)
        grayscale = rgb2gray(frame)  # Convert to grayscale
        # Plot the grayscale frame
    
        # Convert to grayscale
        grayscale = rgb2gray(frame)
        
        # Normalize the grayscale image to enhance contrast
        normalized_frame = (grayscale - grayscale.min()) / (grayscale.max() - grayscale.min())
        
        # Resize the frame
        resized_frame = resize(normalized_frame, (84, 84), anti_aliasing=True, mode='reflect', preserve_range=True)
        
        return resized_frame
    
    def _render_on_surface(self, surface):
        surface.fill((0, 0, 0))
        pygame.draw.rect(surface, (255, 255, 255), self.left_paddle)
        pygame.draw.rect(surface, (255, 255, 255), self.right_paddle)
        pygame.draw.ellipse(surface, (255, 255, 255), self.ball)

    def _apply_action(self, action):
        if action == 1 and self.left_paddle.top > 0:
            self.left_paddle.y -= self.paddle_speed
        elif action == 2 and self.left_paddle.bottom < self.height:
            self.left_paddle.y += self.paddle_speed

    def ai_move(self):
        if self.ai_delay_counter % self.ai_reaction_delay == 0:
            if self.ball.y < self.right_paddle.y + self.paddle_height / 2 and self.right_paddle.top > 0:
                self.right_paddle.y -= self.ai_paddle_speed
            if self.ball.y > self.right_paddle.y + self.paddle_height / 2 and self.right_paddle.bottom < self.height:
                self.right_paddle.y += self.ai_paddle_speed
        self.ai_delay_counter += 1

    def _update_game_state(self):
        reward = 0  # Initialize reward

        # Update ball position
        self.ball.x += self.ball_speed_x
        self.ball.y += self.ball_speed_y

        # Ball collision with walls
        if self.ball.y <= 0 or self.ball.y >= self.height - self.ball_size:
            self.ball_speed_y *= -1

        # Ball collision with paddles
        if self.ball.colliderect(self.right_paddle):
            if self.ball_speed_x > 0:  # Ensure collision only from left side of right paddle
                self.ball_speed_x *= -1
                hit_pos = (self.ball.centery - self.right_paddle.centery) / (self.paddle_height / 2)
                self.ball_speed_y = hit_pos * abs(self.ball_speed_x)  # Simple deflection logic
                reward = 0.5  # Reward for hitting the paddle
                # Move the ball out of the paddle's collision box
                self.ball.right = self.right_paddle.left - 1

        if self.ball.colliderect(self.left_paddle):
            if self.ball_speed_x < 0:  # Ensure collision only from right side of left paddle
                self.ball_speed_x *= -1
                hit_pos = (self.ball.centery - self.left_paddle.centery) / (self.paddle_height / 2)
                self.ball_speed_y = hit_pos * abs(self.ball_speed_x)  # Simple deflection logic
                reward = 0.5  # Reward for hitting the paddle
                # Move the ball out of the paddle's collision box
                self.ball.left = self.left_paddle.right + 1

        # Check if the ball goes out of bounds
        done = False
        if self.ball.left <= 0:
            self.ball_reset()
            self.right_player_score += 1
            reward = -10  # More significant negative reward for losing the ball
        elif self.ball.right >= self.width:
            self.ball_reset()
            self.left_player_score += 1
            reward = 10  # More significant positive reward for scoring
        done = self._check_done()
        # Ensure the ball's speed remains constant
        speed = np.sqrt(self.ball_speed_x**2 + self.ball_speed_y**2)
        self.ball_speed_x = (self.ball_speed_x / speed) * self.ball_speed
        self.ball_speed_y = (self.ball_speed_y / speed) * self.ball_speed

        return reward, done

    def ball_reset(self):
        self.ball.x = self.width // 2 - self.ball_size // 2
        self.ball.y = self.height // 2 - self.ball_size // 2
        # Ensure the angle is not too close to 0 or 90 degrees
        angle = random.uniform(-np.pi / 4, np.pi / 4)  # Random angle between -45 and 45 degrees
        self.ball_speed_x = self.ball_speed * np.cos(angle) * random.choice([1, -1])
        self.ball_speed_y = self.ball_speed * np.sin(angle) * random.choice([1, -1])

    def get_graph_data(self):
        objects = {
            "ball": [self.ball.x, self.ball.y, self.ball_speed_x, self.ball_speed_y, 1, 0, 0],
            "left_paddle": [self.left_paddle.x, self.left_paddle.y, 0, 0, 0, 1, 0],
            "right_paddle": [self.right_paddle.x, self.right_paddle.y, 0, 0, 0, 1, 0],
        }

        node_features = [features for features in objects.values()]
        x = torch.tensor(node_features, dtype=torch.float32)

        return x 

    def _check_done(self):
        if self.left_player_score >= 10 or self.right_player_score >= 10:
            return True
        return False

if __name__ == "__main__":
    # env_configs = [
    #     {"render_mode": None , "observation_type": "pixel", "paddle_width": 5, "ball_speed": 4},
    #     # Add other c
    #     # igurations if needed
    # ]
    
    env = Monitor(PongEnvNew(render_mode='human', observation_type='graph'))
    obs, _ = env.reset()
    print(f"Observation shape: {obs.shape}")
    total_reward = 0
    done = False
    #model = PPO.load("ppo_pong_pixel")
    model = PPO.load("freeway-CNN-eval.zip")
    #result = evaluate_policy(model,env, n_eval_episodes=1)
    #print(result)
    total_reward = 0
    obs, _ = env.reset()
    while not done:
        action,_ = model.predict(obs)
        #action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        env.render()
    print(f"Total reward: {total_reward}")
