import pygame

class Ball:
    def __init__(self, screen, x=0, y=500, move_dist=5):
        self.screen = screen
        self.radius = 10  # Ball radius
        self.color = (255, 255, 255)  # White color
        self.move_dist = move_dist
        self.rect = pygame.Rect(x + self.screen.get_width() / 2 - self.radius,
                                y + self.screen.get_height() / 2 - self.radius,
                                self.radius * 2, self.radius * 2)
        self.x_move_dist = move_dist
        self.y_move_dist = move_dist

    def move(self):
        self.rect.x += self.x_move_dist
        self.rect.y += self.y_move_dist

    def bounce(self, x_bounce=False, y_bounce=False):
        if x_bounce:
            self.x_move_dist *= -1
        if y_bounce:
            self.y_move_dist *= -1

    def reset(self):
        # Center the ball in the middle of the screen
        self.rect.x = self.screen.get_width() / 2 - self.radius
        self.rect.y = self.screen.get_height() / 2 - self.radius
        self.y_move_dist = self.move_dist  # Reset movement direction

    def draw(self):
        # Draw the ball as a circle
        pygame.draw.ellipse(self.screen, self.color, self.rect)
