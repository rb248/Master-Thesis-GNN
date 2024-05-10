import turtle as tr
from paddle import Paddle
from ball import Ball
from scoreboard import Scoreboard
from ui import UI
from bricks import Bricks
import time


import pygame
import sys

# Initialize Pygame
pygame.init()

# Set up the display
screen_width = 1200
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('Breakout')

# Initialize Pygame


screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('Breakout')
pygame.init()

# Clock to manage game speed
clock = pygame.time.Clock()

# Initialize game components
ui = UI(screen)
ui.header()

scoreboard = Scoreboard(screen, lives=5)
paddle = Paddle(screen)
bricks = Bricks(screen)
ball = Ball(screen)
# Clock to manage game speed
clock = pygame.time.Clock()


game_paused = False
playing_game = True


def pause_game():
    global game_paused
    if game_paused:
        game_paused = False
    else:
        game_paused = True


def check_collision_with_walls(ball, score, ui, playing_game):
    # detect collision with left and right walls:
    if ball.rect.left <= 0 or ball.rect.right >= screen_width:
        ball.bounce(x_bounce=True, y_bounce=False)

    # detect collision with upper wall
    if ball.rect.top <= 0:
        ball.bounce(x_bounce=False, y_bounce=True)

    # detect collision with bottom wall
    if ball.rect.bottom >= screen_height:
        ball.reset()
        score.decrease_lives()
        if score.lives == 0:
            score.reset()
            playing_game = False
            ui.game_over(win=False)
        else:
            ui.change_color()


def check_collision_with_paddle(ball, paddle):
    if ball.rect.colliderect(paddle.rect):
        # Determine the collision side and bounce accordingly
        center_ball = ball.rect.centerx
        center_paddle = paddle.rect.centerx

        if center_ball < center_paddle:  # Ball hits the left side of the paddle
            ball.bounce(x_bounce=True, y_bounce=True)
        elif center_ball > center_paddle:  # Ball hits the right side of the paddle
            ball.bounce(x_bounce=True, y_bounce=True)
        else:
            # Ball hits the middle of the paddle
            ball.bounce(x_bounce=False, y_bounce=True)


def check_collision_with_bricks(ball, score, bricks):
    for brick in bricks.bricks:
        if ball.rect.colliderect(brick.rect):
            score.increase_score()
            brick.quantity -= 1
            if brick.quantity == 0:
                bricks.bricks.remove(brick)
            # Determine collision direction
            # Note: Simple version without precise side detection
            ball.bounce(x_bounce=False, y_bounce=True)
            break


# Game state variables
game_paused = False
playing_game = True


def pause_game():
    global game_paused
    game_paused = not game_paused
    if game_paused:
        ui.paused_status()
    else:
        ui.header()


# Main game loop
while playing_game:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            playing_game = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                pause_game()

    # Continuous key checks for paddle movement
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        paddle.move_left()
    if keys[pygame.K_RIGHT]:
        paddle.move_right()

    if not game_paused:
        screen.fill((0, 0, 0))  # Clear the screen

        # Move and draw game elements
        
        paddle.draw()
        bricks.draw()
        scoreboard.draw()
        
        ball.draw()
        ball.move()
        # Collision detection
        # Assuming this method also handles the bounce
        check_collision_with_walls(ball, scoreboard, ui, playing_game)

        if scoreboard.lives == 0:
            scoreboard.reset()
            ui.game_over(False)
            playing_game = False

        # Method to handle ball-paddle collision
        check_collision_with_paddle(ball, paddle)

        # Method handling ball-brick collisions
        check_collision_with_bricks(ball, scoreboard, bricks)

        # Check game win condition
        if len(bricks.bricks) == 0:
            ui.game_over(True)
            playing_game = False

        pygame.display.flip()  # Update the screen
    clock.tick(60)  # Limit the frame rate to 60 frames per second

pygame.quit()
sys.exit()