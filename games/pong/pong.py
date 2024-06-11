import pygame
import random
#import pong_features
# Initialize Pygame
pygame.init()
# ai reaction time
ai_reaction_time = 2  # milliseconds
ai_last_reaction_time = pygame.time.get_ticks()
# Set up the display
screen_width, screen_height = 800, 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Pong")
grid_width, grid_height = 10, 8
#node_width, node_height = screen_width // grid_width, screen_height // grid_height
#nodes = [[{'features': [0]*18} for _ in range(grid_height)] for _ in range(grid_width)]
#G = pong_features.create_initial_graph(grid_width, grid_height)

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Paddle properties
paddle_width, paddle_height = 10, 40
paddle_speed = 5
left_paddle = pygame.Rect(50, screen_height // 2 - paddle_height // 2, paddle_width, paddle_height)
right_paddle = pygame.Rect(screen_width - 50 - paddle_width, screen_height // 2 - paddle_height // 2, paddle_width, paddle_height)

# Ball properties
ball_size = 15
ball_speed_x, ball_speed_y = 4 * random.choice((1, -1)), 4 * random.choice((1, -1))
ball = pygame.Rect(screen_width // 2 - ball_size // 2, screen_height // 2 - ball_size // 2, ball_size, ball_size)

# Player Lives
left_player_lives = 3
right_player_lives = 3

# Font for displaying lives
font = pygame.font.Font(None, 36)

# Game loop
running = True
clock = pygame.time.Clock()

while running:
    current_time = pygame.time.get_ticks()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Paddle movement
    left_paddle_move = 0
    right_paddle_move = 0
    keys = pygame.key.get_pressed()
    if keys[pygame.K_w] and left_paddle.top > 0:
        left_paddle.y -= paddle_speed 
        left_paddle_move = -1
    if keys[pygame.K_s] and left_paddle.bottom < screen_height:
        left_paddle.y += paddle_speed
        left_paddle_move = 1
    
    # Right paddle movement
    if current_time - ai_last_reaction_time > ai_reaction_time:
        if ball.y < right_paddle.y + paddle_height / 2 and right_paddle.top > 0:
            right_paddle.y -= paddle_speed
        if ball.y > right_paddle.y + paddle_height / 2 and right_paddle.bottom < screen_height:
            right_paddle.y += paddle_speed
        ai_last_reaction_time = current_time

    # Ball movement
    ball.x += ball_speed_x
    ball.y += ball_speed_y

    # Ball collision with top and bottom
    if ball.top <= 0 or ball.bottom >= screen_height:
        ball_speed_y *= -1

    # Ball collision with paddles
    if ball.colliderect(left_paddle) or ball.colliderect(right_paddle):
        ball_speed_x *= -1

    # Ball out of bounds
    if ball.left <= 0:
        left_player_lives -= 1
        ball.x, ball.y = screen_width // 2 - ball_size // 2, screen_height // 2 - ball_size // 2
        ball_speed_x, ball_speed_y = 4, 4 * random.choice((1, -1))
    if ball.right >= screen_width:
        right_player_lives -= 1
        ball.x, ball.y = screen_width // 2 - ball_size // 2, screen_height // 2 - ball_size // 2
        ball_speed_x, ball_speed_y = -4, 4 * random.choice((1, -1))

    
    # Game over check
    if left_player_lives == 0 or right_player_lives == 0:
        running = False  # Stop the game

    # Drawing
    screen.fill(BLACK)
    pygame.draw.rect(screen, WHITE, left_paddle)
    pygame.draw.rect(screen, WHITE, right_paddle)
    pygame.draw.ellipse(screen, WHITE, ball)
    pygame.draw.aaline(screen, WHITE, (screen_width // 2, 0), (screen_width // 2, screen_height))

    # Display lives
    left_lives_text = font.render(f'Lives: {left_player_lives}', True, WHITE)
    right_lives_text = font.render(f'Lives: {right_player_lives}', True, WHITE)
    screen.blit(left_lives_text, (10, 10))
    screen.blit(right_lives_text, (screen_width - 150, 10))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
