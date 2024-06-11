import pygame
import random

pygame.init()

# Game window dimensions
WIDTH, HEIGHT = 800, 600
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Shooting Game")

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

# Shooter settings
shooter_pos = [WIDTH // 2, HEIGHT - 50]
shooter_speed = 5

# Target settings
target_pos = [random.randint(20, WIDTH - 20), random.randint(20, HEIGHT / 2)]
target_speed = 2
target_direction = 1

# Bullet settings
bullet_pos = []
bullet_speed = 10

clock = pygame.time.Clock()

def draw_game():
    win.fill(BLACK)
    pygame.draw.circle(win, RED, target_pos, 20)
    pygame.draw.rect(win, WHITE, (*shooter_pos, 50, 20))
    for bullet in bullet_pos:
        pygame.draw.rect(win, WHITE, (*bullet, 10, 5))
    pygame.display.update()

def move_target():
    global target_direction
    target_pos[0] += target_speed * target_direction
    if target_pos[0] >= WIDTH - 20 or target_pos[0] <= 20:
        target_direction *= -1

def shoot_bullet():
    bullet_pos.append([shooter_pos[0] + 20, shooter_pos[1]])

def move_bullets():
    for bullet in bullet_pos[:]:
        bullet[1] -= bullet_speed
        if bullet[1] < 0:
            bullet_pos.remove(bullet)

def check_collision():
    global shooter_pos, target_pos
    for bullet in bullet_pos:
        if target_pos[0] - 20 < bullet[0] < target_pos[0] + 20 and target_pos[1] - 20 < bullet[1] < target_pos[1] + 20:
            shooter_pos = [random.randint(50, WIDTH - 50), HEIGHT - 50]
            target_pos = [random.randint(20, WIDTH - 20), random.randint(20, HEIGHT / 2)]
            bullet_pos.remove(bullet)

running = True
while running:
    clock.tick(60)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                shoot_bullet()

    move_target()
    move_bullets()
    check_collision()
    draw_game()

pygame.quit()
