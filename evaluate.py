from pathlib import Path
import sys
ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))


import pygame
from utils.settings import GRID_SIZE, CELL_SIZE
from environment.environment import Environment
import numpy as np

class Game:
    def __init__(self, fps, env: Environment):
        pygame.init()
        self.fps = fps
        self.env = env
        self.width = GRID_SIZE[0] * CELL_SIZE
        self.height = GRID_SIZE[1] * CELL_SIZE
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font("assets/PressStart2P-Regular.ttf", 25)
        self.score = 0

    def step(self, action):
        pass

    def game_draw(self):
        # Draw snake
        for x in self.env.snake.body:
            pygame.draw.rect(self.screen, (255, 0, 0), (*x, CELL_SIZE, CELL_SIZE))
            pygame.draw.rect(self.screen, (255, 255, 255), (*x, CELL_SIZE, CELL_SIZE), 1)

        # Draw food
        if Game.__flag:
            pygame.draw.rect(
                self.screen, (255, 255, 0),
                (*self.env.food.position, CELL_SIZE, CELL_SIZE),
                border_radius=15
            )
            Game.__flag = False
        else:
            pygame.draw.rect(
                self.screen, (255, 255, 0),
                (*(self.env.food.position + np.array([5, 5])), CELL_SIZE - 10, CELL_SIZE - 10),
                border_radius=15
            )
            Game.__flag = True

        # Draw score
        try:
            if not os.path.exists("assets/PressStart2P-Regular.ttf"):
                raise FileNotFoundError("Can not found: {}".format("assets/PressStart2P-Regular.ttf"))
        except FileNotFoundError as error:
            print(error)
            self.font = pygame.font.Font(None, 25)
        text = self.font.render("SCORE: {:04d}".format(self.score), True, (0, 255, 0))
        self.screen.blit(text, dest=(15, 15))

        pygame.display.flip()
        self.clock.tick(self.fps)
        self.screen.fill(self.background_color)

    def is_running(self):
        if not self.env.done:
            return False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True
