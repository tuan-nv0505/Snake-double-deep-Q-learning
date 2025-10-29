import os.path
from collections import deque
from pathlib import Path
import sys
ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

from utils.direction import Direction
from agent.deep_q_network import DeepQNetwork
import pygame
from utils.utils import get_args
from environment.environment import Environment
import numpy as np
import torch
import sys

args = get_args()
class Game:
    def __init__(self, fps, env: Environment):
        pygame.init()
        self.fps = fps
        self.env = env
        self.width = self.env.grid_size[1] * args.cell_size
        self.height = self.env.grid_size[0] * args.cell_size
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
                frame = self.env.reset()
                stack_frames = deque([torch.from_numpy(frame)] * args.frame_size, maxlen=args.frame_size)

                while not self.env.done:
                    with torch.no_grad():
                        state = torch.stack(list(stack_frames))
                        q_values = agent(state.unsqueeze(0))
                        action = torch.argmax(q_values).item()
                        next_frame = self.env.step(action)[0]
                        stack_frames.append(torch.from_numpy(next_frame))
                        self.draw()
        pygame.quit()
        sys.exit()


    def draw(self):
        # Draw snake
        for element in self.env.snake.position:
            y, x = element * args.cell_size
            pygame.draw.rect(self.screen, (255, 0, 0), (x, y, args.cell_size, args.cell_size))
            pygame.draw.rect(self.screen, (255, 255, 255), (x, y, args.cell_size, args.cell_size), 1)

        # Draw food
        if not self.scale_food:
            y, x = self.env.food.position * args.cell_size
            pygame.draw.rect(
                self.screen, (255, 255, 0),
                (x, y, args.cell_size, args.cell_size),
                border_radius=15
            )
            self.scale_food = True
        else:
            y, x = self.env.food.position * args.cell_size
            pygame.draw.rect(
                self.screen, (255, 255, 0),
                (*(np.array([x, y]) + np.array([5, 5])), args.cell_size - 10, args.cell_size - 10),
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
    path_checkpoint = 'checkpoint/snake_dqn.pth'
    agent = None
    if os.path.exists(path_checkpoint):
        print('Load file snake_dqn.pth successfully. Start evaluating...')
        agent = DeepQNetwork(3)
        agent.load_state_dict(torch.load(path_checkpoint))

    game = Game(fps=30, env=Environment(args.grid_size, 0))
    game.play(agent=agent)