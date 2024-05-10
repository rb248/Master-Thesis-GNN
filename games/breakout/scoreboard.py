import pygame

class Scoreboard:
    def __init__(self, screen, lives, font=('arial', 18), color=(255, 255, 255)):
        self.screen = screen
        self.font = pygame.font.SysFont(*font)
        self.color = color
        self.lives = lives
        self.score = 0
        self.highScore = self.load_high_score()
        self.rect = pygame.Rect(10, 10, 300, 40)  # Adjust size as needed
        self.update_score()

    def load_high_score(self):
        try:
            with open('highestScore.txt', 'r') as file:
                return int(file.read())
        except (FileNotFoundError, ValueError):
            with open('highestScore.txt', 'w') as file:
                file.write('0')
            return 0

    def update_score(self):
        self.text = f"Score: {self.score} | Highest Score: {self.highScore} | Lives: {self.lives}"
        self.text_surface = self.font.render(self.text, True, self.color)

    def increase_score(self):
        self.score += 1
        if self.score > self.highScore:
            self.highScore = self.score
        self.update_score()

    def decrease_lives(self):
        self.lives -= 1
        self.update_score()

    def reset(self):
        self.score = 0
        self.update_score()
        with open('highestScore.txt', 'w') as file:
            file.write(str(self.highScore))

    def draw(self):
        self.screen.blit(self.text_surface, (self.rect.x, self.rect.y))
