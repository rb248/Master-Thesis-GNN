import gym
from gym import spaces
import numpy as np
import pygame
from paddle import Paddle
from ball import Ball
from scoreboard import Scoreboard
from ui import UI
from bricks import Bricks
import time

class BreakoutEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(BreakoutEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(3)  # actions: move left, stay, move right
        # Example for observation space: the game state
        self.observation_space = spaces.Box(low=0, high=255, shape=(600, 1200, 3), dtype=np.uint8)
        self.screen_width = 1200
        self.screen_height = 600
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption('Breakout')
        pygame.init()
        self.clock = pygame.time.Clock()

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

        return self.get_state()    

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

        # Draw all game elements
        self.paddle.draw()
        self.ball.draw()
        self.bricks.draw()
        self.scoreboard.draw()

        # Check if game over
        done = self.scoreboard.lives == 0 or len(self.bricks.bricks) == 0

        # Update the display to show the new positions of game elements
        pygame.display.flip()

        # Additional info can be passed, though not used here
        info = {}

        return self.get_state(), reward, done, info


    

    def render(self, mode='human'):

        pass # update turtle graphics if needed

    def close(self):
        tr.bye()

    def check_collisions(self):
        # Implement collision checks
        reward = 0
        # Implement collision logic with walls, paddle, bricks
        # Adjust reward accordingly
        return reward

    def get_state(self):
        # Implement a method to extract the current game state
        # This might include positions of the paddle, ball, bricks, etc.
        return pygame.surfarray.array3d(pygame.display.get_surface())
    
if __name__ == "__main__":
    env = BreakoutEnv()
    env.reset()
    for _ in range(1000):
        env.step(env.action_space.sample())
        time.sleep(0.1)
    env.close()
