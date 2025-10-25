from pathlib import Path
import sys

from utils.direction import Direction

ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

from agent.deep_q_network import DeepQNetwork
import pygame
from utils.settings import GRID_SIZE, CELL_SIZE
from environment.environment import Environment
import numpy as np
import torch
from torchvision.transforms import ToTensor
import sys

class Game:
    def __init__(self, fps, env: Environment):
        pygame.init()
        self.fps = fps
        self.env = env
        self.width = self.env.grid_size[1] * CELL_SIZE
        self.height = self.env.grid_size[0] * CELL_SIZE
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font("assets/PressStart2P-Regular.ttf", 25)
        self.score = 0
        self.scale_food = False

    def play(self, agent=None):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            if agent is None:
                self.env.reset()
                while not self.env.done:
                    pygame.event.pump()
                    keys = pygame.key.get_pressed()
                    get_action = lambda new_direction, current_direction: (new_direction.value - current_direction.value + 1 + 4) % 4
                    action = 1
                    if keys[pygame.K_UP] and self.env.snake.direction != Direction.DOWN:
                        action = get_action(Direction.UP, self.env.snake.direction)
                    elif keys[pygame.K_DOWN] and self.env.snake.direction != Direction.UP:
                        action = get_action(Direction.DOWN, self.env.snake.direction)
                    elif keys[pygame.K_LEFT] and self.env.snake.direction != Direction.RIGHT:
                        action = get_action(Direction.LEFT, self.env.snake.direction)
                    elif keys[pygame.K_RIGHT] and self.env.snake.direction != Direction.LEFT:
                        action = get_action(Direction.RIGHT, self.env.snake.direction)

                    self.env.step(action)
                    self.draw()
            else:
                state = ToTensor()(self.env.reset())
                while not self.env.done:
                    with torch.no_grad():
                        q_values = agent(state.unsqueeze(0))
                        action = torch.argmax(q_values).item()
                        state = ToTensor()(self.env.step(action)[0])
                        self.draw()
        pygame.quit()
        sys.exit()


    def draw(self):
        # Draw snake
        for element in self.env.snake.position:
            y, x = element * CELL_SIZE
            pygame.draw.rect(self.screen, (255, 0, 0), (x, y, CELL_SIZE, CELL_SIZE))
            pygame.draw.rect(self.screen, (255, 255, 255), (x, y, CELL_SIZE, CELL_SIZE), 1)

        # Draw food
        if not self.scale_food:
            y, x = self.env.food.position * CELL_SIZE
            pygame.draw.rect(
                self.screen, (255, 255, 0),
                (x, y, CELL_SIZE, CELL_SIZE),
                border_radius=15
            )
            self.scale_food = True
        else:
            y, x = self.env.food.position * CELL_SIZE
            pygame.draw.rect(
                self.screen, (255, 255, 0),
                (*(np.array([x, y]) + np.array([5, 5])), CELL_SIZE - 10, CELL_SIZE - 10),
                border_radius=15
            )
            self.scale_food = False

        # Draw score
        text = self.font.render("SCORE: {:04d}".format(self.score), True, (0, 255, 0))
        self.screen.blit(text, dest=(15, 15))

        pygame.display.flip()
        self.clock.tick(self.fps)
        self.screen.fill((0, 0, 0))

if __name__ == '__main__':
    agent = DeepQNetwork(3)
    agent.load_state_dict(torch.load('agent/double_dqn_snake.pth'))
    game = Game(fps=30, env=Environment(GRID_SIZE, 0))
    game.play(agent=agent)