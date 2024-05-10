import pygame
import random

# Font and color definitions
FONT = ('couriernew', 52)  # Pygame does not support the "normal" weight attribute
FONT2 = ('couriernew', 32)
ALIGNMENT = 'center'
COLOR_LIST = [
    (173, 216, 230), (65, 105, 225), (176, 196, 222), (70, 130, 180),
    (224, 255, 255), (135, 206, 250), (238, 130, 238), (250, 128, 114),
    (255, 99, 71), (244, 164, 96), (128, 0, 128), (255, 20, 147),
    (60, 179, 113), (240, 230, 140)
]

class UI:
    def __init__(self, screen):
        self.screen = screen
        self.font = pygame.font.SysFont(*FONT)
        self.font2 = pygame.font.SysFont(*FONT2)
        self.color = random.choice(COLOR_LIST)

    def header(self):
        self.clear()
        header_text = 'Breakout'
        instructions_text = 'Press Space to PAUSE or RESUME the Game'
        self.render_text(header_text, self.screen.get_width() // 2, 100, self.font)
        self.render_text(instructions_text, self.screen.get_width() // 2, 150, pygame.font.SysFont('calibri', 14))

    def change_color(self):
        self.color = random.choice(COLOR_LIST)
        self.header()

    def paused_status(self):
        self.header()

    def game_over(self, win):
        self.clear()
        if win:
            message = 'You Cleared the Game'
        else:
            message = "Game is Over"
        self.render_text(message, self.screen.get_width() // 2, self.screen.get_height() // 2, self.font)

    def render_text(self, text, x, y, font):
        text_surface = font.render(text, True, self.color)
        text_rect = text_surface.get_rect(center=(x, y))
        self.screen.blit(text_surface, text_rect)

    def clear(self):
        # This function would need to clear only the areas where text was displayed if needed
        pass
