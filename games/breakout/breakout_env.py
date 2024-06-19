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
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, render_mode='human', observation_type='pixel', num_frames=4):
        super(BreakoutEnv, self).__init__()

        self.render_mode = render_mode
        self.observation_type = observation_type
        self.num_frames = num_frames

        # Action space (move left, stay, move right)
        self.action_space = spaces.Discrete(3)

        # Observation space
        self.window_width = 1200
        self.window_height = 600
        if observation_type == 'pixel':
            self.observation_space = spaces.Box(low=0, high=255,
                                                shape=(self.window_height, self.window_width, 3 * self.num_frames),
                                                dtype=np.uint8)
        else:
            brick_width = 60
            brick_spacing = 5
            num_bricks_per_lane = (self.screen.get_width() - 2 * brick_spacing) // (brick_width + brick_spacing)
            self.total_bricks = 5 * num_bricks_per_lane

            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.total_bricks+ 2, 7), dtype=np.float32)

        # Initialize the game
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Breakout")
        self.frame_buffer = np.zeros((self.window_height, self.window_width, 3 * self.num_frames), dtype=np.uint8)
        self.clock = pygame.time.Clock()

        self.reset()

    def reset(self):
        self.paddle = Paddle(self.screen)
        self.ball = Ball(self.screen)
        self.bricks = Bricks(self.screen)
        self.scoreboard = Scoreboard(self.screen, lives=5)
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
        if self.observation_type == 'pixel':
            return self.frame_buffer
        else:
            return self.get_object_data()

    def step(self, action):
        assert self.action_space.contains(action), f"{action} is an invalid action"
        
        self.screen.fill((0, 0, 0))
        
        if action == 0:
            self.paddle.move_left()
        elif action == 1:
            self.paddle.move_right()
        elif action == 2:
            pass  # Do nothing for 'stay' action

        self.ball.move()
        
        reward = 0
        reward += self.check_collision_with_walls(self.ball, self.scoreboard, self.ui)
        reward += self.check_collision_with_paddle(self.ball, self.paddle)
        reward += self.check_collision_with_bricks(self.ball, self.scoreboard, self.bricks)
        
        new_frame = self.get_state()
        self.update_frame_buffer(new_frame)
        
        self.paddle.draw()
        self.ball.draw()
        self.bricks.draw()
        self.scoreboard.draw()
        
        done = self.scoreboard.lives == 0 or len(self.bricks.bricks) == 0

        pygame.display.flip()

        if self.observation_type == 'pixel':
            observation = self.frame_buffer
        else:
            observation = self.get_object_data()

        return observation, reward, done, {}

    def check_collision_with_walls(self, ball, score, ui):
        reward = 0
        if ball.rect.left <= 0 or ball.rect.right >= self.window_width:
            ball.bounce(x_bounce=True, y_bounce=False)

        if ball.rect.top <= 0:
            ball.bounce(x_bounce=False, y_bounce=True)

        if ball.rect.bottom >= self.window_height:
            ball.reset()
            reward = -100
            score.decrease_lives()
            
            if score.lives == 0:
                score.reset()
                ui.game_over(win=False)
            else:
                ui.change_color()
        return reward

    def check_collision_with_paddle(self, ball, paddle):
        reward = 0
        if ball.rect.colliderect(paddle.rect):
            center_ball = ball.rect.centerx
            center_paddle = paddle.rect.centerx
            reward = 10

            if center_ball < center_paddle:
                ball.bounce(x_bounce=True, y_bounce=True)
            elif center_ball > center_paddle:
                ball.bounce(x_bounce=True, y_bounce=True)
            else:
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
                ball.bounce(x_bounce=False, y_bounce=True)
                break
        return reward

    def get_state(self):
        surface_array = pygame.surfarray.array3d(pygame.display.get_surface())
        transposed_array = np.transpose(surface_array, axes=(1, 0, 2))
        return transposed_array

    def update_frame_buffer(self, new_frame):
        self.frame_buffer = np.roll(self.frame_buffer, shift=-3, axis=2)
        self.frame_buffer[:, :, -3:] = new_frame

    def get_object_data(self):
        object_features = []

        ball_features = [self.ball.rect.x, self.ball.rect.y, self.ball.x_move_dist, self.ball.y_move_dist, 1, 0, 0]
        object_features.append(ball_features)

        paddle_features = [self.paddle.rect.x, self.paddle.rect.y, 0, 0, 0, 1, 0]
        object_features.append(paddle_features)

        # left_wall_features = [0, 0, 0, 0, 0, 0, 1]
        # object_features.append(left_wall_features)
        # right_wall_features = [self.window_width, 0, 0, 0, 0, 0, 1]
        # object_features.append(right_wall_features)
        # top_wall_features = [0, 0, 0, 0, 0, 0, 1]
        brick_count = 0
        for brick in self.bricks.bricks:
            brick_features = [brick.rect.x, brick.rect.y, 0, 0, 0, 0, 1]
            object_features.append(brick_features) 
            brick_count += 1
        
        while brick_count < self.total_bricks:
            brick_features = [0, 0, 0, 0, 0, 0, 0]
            object_features.append(brick_features)
            brick_count += 1

        return torch.tensor(object_features, dtype=torch.float32)

    def render(self, mode='human'):
        if mode == 'human':
            pygame.display.flip()
        elif mode == 'rgb_array':
            return pygame.surfarray.array3d(pygame.display.get_surface())

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    env = BreakoutEnv()
    obs = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()
        if reward < 0:
            print("Ball lost")
        if done:
            print("Game Over. Restarting...")
            obs = env.reset()
            done = False

    env.close()