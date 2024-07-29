import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import random
from skimage.transform import resize
from stable_baselines3 import PPO
from collections import deque

class PongEnvNew(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60, "observation_types": ["pixel", "graph"]}

    def __init__(self, render_mode=None, observation_type='pixel', paddle_width=5, paddle_height=20, ball_size=5, paddle_speed=10, ai_paddle_speed=10, frame_stack=4):
        super(PongEnvNew, self).__init__()
        self.width = 210
        self.height = 160
        self.render_mode = render_mode
        self.observation_type = observation_type
        self.paddle_width = paddle_width
        self.paddle_height = paddle_height
        self.ball_size = ball_size
        self.paddle_speed = paddle_speed
        self.ai_paddle_speed = ai_paddle_speed  # AI paddle speed
        self.frame_stack = frame_stack  # Number of frames to stack
        self.action_space = spaces.Discrete(3)  # [Stay, Up, Down]

        if observation_type == "pixel":
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.frame_stack, 84, 84), dtype=np.uint8)
        else:
            # Define a generic observation space for graph data
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3, 7), dtype=np.float32)  # Number of objects and feature length

        self.screen = None  # Delay screen creation until rendering
        self.clock = None
        self.ai_reaction_time = 5  # milliseconds
        self.frame_buffer = deque(maxlen=self.frame_stack)
    
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        if seed is not None:
            self.seed(seed)  # Seed the RNG for the environment
        self.ball = pygame.Rect(self.width // 2 - self.ball_size // 2, self.height // 2 - self.ball_size // 2, self.ball_size, self.ball_size)
        self.left_paddle = pygame.Rect(20, self.height // 2 - self.paddle_height // 2, self.paddle_width, self.paddle_height)
        self.right_paddle = pygame.Rect(self.width - 20 - self.paddle_width, self.height // 2 - self.paddle_height // 2, self.paddle_width, self.paddle_height)
        self.ai_last_reaction_time = pygame.time.get_ticks()
        
        # Set random speeds for the ball
        angle = random.uniform(-np.pi / 4, np.pi / 4)  # Random angle between -45 and 45 degrees
        speed = 2  # Constant speed
        self.ball_speed_x = speed * np.cos(angle) * random.choice([1, -1])
        self.ball_speed_y = speed * np.sin(angle) * random.choice([1, -1])
        
        self.left_player_score = 0
        self.right_player_score = 0
        self.frame_buffer.clear()
        
        # Fill the frame buffer with the initial frame
        if self.observation_type == "pixel":
            obs = self._get_obs()
            for _ in range(self.frame_stack):
                self.frame_buffer.append(obs)
            return np.array(self.frame_buffer), {}
        else:
            return self.get_graph_data(), {}

    def step(self, action):
        if not pygame.display.get_init():
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.display.quit()
                pygame.quit()
                raise SystemExit("Pygame QUIT event received.")
        self.ai_move()
        self._apply_action(action)
        self._update_game_state()
        
        reward = 0
        done = False
        if self.ball.left <= 0 or self.ball.right >= self.width:
            reward = 1 if self.ball.left <= 0 else -1
            done = True

        # Increment step counter
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True

        # Step penalty to encourage faster completion
        reward -= 0.01

        # Update observation
        obs = self._get_obs()
        self.frame_buffer.append(obs)
        info = {}
        truncated = False

        return np.array(self.frame_buffer), reward, done, truncated, info

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
        if self.render_mode == "rgb_array":
            self._render_on_surface(self.offscreen_surface)
            frame = pygame.surfarray.array3d(self.offscreen_surface)
        else:
            self._render_on_surface(self.screen)
            frame = pygame.surfarray.array3d(self.screen)
        
        frame = np.transpose(frame, (1, 0, 2))  # Transpose to match (height, width, channels)
        grayscale = np.dot(frame[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)  # Convert to grayscale
        resized_frame = resize(grayscale, (84, 84), anti_aliasing=True, preserve_range=True).astype(np.uint8)  # Resize and update the last frame
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
        current_time = pygame.time.get_ticks()
        # AI paddle movement
        if current_time - self.ai_last_reaction_time > self.ai_reaction_time:
            if self.ball.y < self.right_paddle.y + self.paddle_height / 2 and self.right_paddle.top > 0:
                self.right_paddle.y -= self.ai_paddle_speed
            if self.ball.y > self.right_paddle.y + self.paddle_height / 2 and self.right_paddle.bottom < self.height:
                self.right_paddle.y += self.ai_paddle_speed
            self.ai_last_reaction_time = current_time

    def _update_game_state(self):
        self.ball.x += self.ball_speed_x
        self.ball.y += self.ball_speed_y

        # Check for collisions with top and bottom of the screen
        if self.ball.top <= 0:
            self.ball.top = 0
            self.ball_speed_y *= -1
        elif self.ball.bottom >= self.height:
            self.ball.bottom = self.height
            self.ball_speed_y *= -1

        # Check if the ball goes out of bounds
        if self.ball.left <= 0:
            self.ball_reset()
            self.right_player_score += 1
        elif self.ball.right >= self.width:
            self.ball_reset()
            self.left_player_score += 1

        # Check for paddle collisions
        if (self.ball.x > self.right_paddle.left and self.ball.x < self.right_paddle.right and 
            self.ball.y > self.right_paddle.top and self.ball.y < self.right_paddle.bottom):
            self.ball.x = self.right_paddle.left - self.ball.width
            angle = random.uniform(-np.pi / 4, np.pi / 4)  # Slight random angle variation
            speed = np.hypot(self.ball_speed_x, self.ball_speed_y)  # Keep the speed constant
            self.ball_speed_x = -speed * np.cos(angle)
            self.ball_speed_y = speed * np.sin(angle)

        if (self.ball.x < self.left_paddle.right and self.ball.x > self.left_paddle.left and 
            self.ball.y > self.left_paddle.top and self.ball.y < self.left_paddle.bottom):
            self.ball.x = self.left_paddle.right
            angle = random.uniform(-np.pi / 4, np.pi / 4)  # Slight random angle variation
            speed = np.hypot(self.ball_speed_x, self.ball_speed_y)  # Keep the speed constant
            self.ball_speed_x = speed * np.cos(angle)
            self.ball_speed_y = speed * np.sin(angle)

    def ball_reset(self):
        self.ball.x = self.width // 2 - self.ball_size // 2
        self.ball.y = self.height // 2 - self.ball_size // 2
        # Ensure the angle is not too close to 0 or 90 degrees
        angle = random.uniform(np.pi / 6, np.pi / 3)  # Random angle between 30 and 60 degrees
        speed = 2  # Constant speed
        self.ball_speed_x = speed * np.cos(angle) * random.choice([1, -1])
        self.ball_speed_y = speed * np.sin(angle) * random.choice([1, -1])

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
        if self.left_player_score >= 20 or self.right_player_score >= 20:
            return True
        return False

if __name__ == "__main__":
    env = make_vec_env(lambda: PongEnvNew(render_mode=None, observation_type='pixel'), n_envs=4, vec_env_cls=SubprocVecEnv)
    model = PPO('CnnPolicy', env, verbose=1)
    model.learn(total_timesteps=100000)
    model.save("ppo_custom_env_pixel")

    # Test the trained model
    env = PongEnvNew(render_mode='human', observation_type='pixel')
    obs, _ = env.reset()
    total_reward = 0
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        env.render()
    print(f"Total reward: {total_reward}")
