from turtle import Turtle


MOVE_DIST = 70


import pygame

class Paddle:
    def __init__(self, screen, width=200, height=20, move_dist=70):
        self.screen = screen
        self.width = width
        self.height = height
        self.move_dist = move_dist
        self.color = (70, 130, 180)  # Steel blue
        self.rect = pygame.Rect((screen.get_width() / 2 - width / 2, 
                                 screen.get_height() - 40, 
                                 width, 
                                 height))

    def draw(self):
        pygame.draw.rect(self.screen, self.color, self.rect)

    def move_left(self):
        # Check if the paddle is not going beyond the left screen edge
        if self.rect.x > 0:
            self.rect.x -= self.move_dist

    def move_right(self):
        # Check if the paddle is not going beyond the right screen edge
        if self.rect.x + self.width < self.screen.get_width():
            self.rect.x += self.move_dist
