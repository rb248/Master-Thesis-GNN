import pygame
import random

# Initialize pygame
pygame.init()
score = 0  # Initialize score
pygame.font.init()  # Initialize font module
font = pygame.font.SysFont(None, 36)  # Create a font object

# Set up the game window
window_width = 800
window_height = 600
window = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("Freeway Game")
# set background image
background_image = pygame.image.load("../images/Atari - background.png")
background_image = pygame.transform.scale(background_image, (window_width, window_height))
# Define colors
white = (255, 255, 255)
black = (0, 0, 0)
# Define the player (chicken)

player_width = 30
player_height = 30
player_x = window_width // 2 - player_width // 2
player_y = window_height - player_height - 10
player_rect = pygame.Rect(player_x, player_y, player_width, player_height)

player_speed = 5
player_image = pygame.image.load("../images/chicken.png").convert_alpha()
car_image = pygame.image.load("../images/car2.png").convert_alpha()
car_width = 50
car_height = 50
car_image = pygame.transform.scale(car_image, (car_width, car_height))
player_image = pygame.transform.scale(player_image, (player_width, player_height))
def draw_player(player_x, player_y):
    window.blit(player_image, (player_rect.x, player_rect.y))  # Draw the image

# Define the cars

car_x = random.randint(0, window_width - car_width)
car_y = 0
car_speed = 3

lanes = [100, 200, 300, 400, 500, 600 , 700]  # Example lane positions on the y-axis
#lanes = [100, 150, 200, 250, 300, 350, 400,450, 500,550, 600, 700]
# Define multiple cars with random starting x positions and lanes
cars = []
number_of_cars = 20

for i in range(number_of_cars):
    car = {
        'x': random.randint(0, window_width),
        'lane': random.choice(lanes),
        'speed': random.randint(2, 5)
    }
    cars.append(car)

def draw_cars():
    for car in cars:
        #car_rect = pygame.Rect(car['x'], car['lane'], car_width, car_height)
        window.blit(car_image, (car['x'], car['lane']))  # Draw each car at its lane

# Game loop

# In the game loop
running = True
clock = pygame.time.Clock()
# Start the game timer
start_time = pygame.time.get_ticks()  # Get the current time
game_duration = 136000  # 2 minutes and 16 seconds in milliseconds

while running:
    elapsed_time = pygame.time.get_ticks() - start_time

    # Check if the time limit is reached
    if elapsed_time >= game_duration:
        # End the game or transition to a different screen
        running = False
    # Move the cars
    for car in cars:
        car['x'] += car['speed']
        # Check if the car has moved off-screen; if so, reset it
        if car['x'] > window_width:
            car['x'] = -random.randint(100, 300)
            car['speed'] = random.randint(2, 5)  # Randomize speed for variety

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP] and player_y > 0:
        player_rect.y -= player_speed
    if keys[pygame.K_DOWN] and player_y < window_height - player_height:
        player_rect.y += player_speed

    window.blit(background_image, (0, 0))  # Draw the background
    draw_player(player_x, player_y)  # Draw the player with updated positions
    draw_cars()  # Draw all the cars
    # Check for collisions
    for car in cars:
        if player_rect.colliderect(pygame.Rect(car['x'], car['lane'], car_width, car_height)):
            score = 0
            player_rect.y = window_height - player_height - 10
            player_rect.x = window_width // 2 - player_width // 2

    # Update the score if the chicken reaches the top of the screen
    if player_rect.y <= 0:
        score += 1
        player_rect.y = window_height - player_height - 10
    
    score_text = font.render(f'Score: {score}', True, white)
    window.blit(score_text, (10, 10))  # Draw the score in the top-left corner
    

    pygame.display.update()
    clock.tick(60)

pygame.quit()