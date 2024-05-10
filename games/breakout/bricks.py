import pygame
import random

COLOR_LIST = [
    (173, 216, 230), (65, 105, 225), (176, 196, 222), (70, 130, 180),
    (224, 255, 255), (135, 206, 250), (238, 130, 238), (250, 128, 114),
    (255, 99, 71), (244, 164, 96), (128, 0, 128), (255, 20, 147),
    (60, 179, 113), (240, 230, 140)
]

weights = [1, 2, 1, 1, 3, 2, 1, 4, 1, 
           3, 1, 1, 1, 4, 1, 3, 2, 2, 
           1, 2, 1, 2, 1, 2, 1]

class Brick:
    def __init__(self, screen, x_cor, y_cor):
        self.screen = screen
        self.color = random.choice(COLOR_LIST)
        self.rect = pygame.Rect(x_cor, y_cor, 60, 30)  # Set brick size
        self.quantity = random.choices(weights, k=1)[0]

    def draw(self):
        pygame.draw.rect(self.screen, self.color, self.rect)

class Bricks:
    def __init__(self, screen):
        self.screen = screen
        self.bricks = []
        self.create_all_lanes()

    def create_lane(self, y_cor):
        # Calculate how many bricks can fit in one row based on screen width
        brick_width = 60
        brick_spacing = 5
        num_bricks = (self.screen.get_width() - 2 * brick_spacing) // (brick_width + brick_spacing)
        start_x = (self.screen.get_width() - (num_bricks * (brick_width + brick_spacing) - brick_spacing)) // 2
        
        for i in range(num_bricks):
            x_cor = start_x + i * (brick_width + brick_spacing)
            brick = Brick(self.screen, x_cor, y_cor)
            self.bricks.append(brick)

    def create_all_lanes(self):
        start_y = 30  # Start 30 pixels down from the top of the screen
        brick_height = 30
        row_spacing = 10  # Space between rows
        num_rows = 5  # Total number of rows
        
        for i in range(num_rows):
            y_cor = start_y + i * (brick_height + row_spacing)
            self.create_lane(y_cor)

    def draw(self):
        for brick in self.bricks:
            brick.draw()