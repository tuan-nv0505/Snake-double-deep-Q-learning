from pathlib import Path
import sys
ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

from utils.direction import Direction
from agent.deep_q_network import DeepQNetwork
import pygame
from utils.utils import get_args
from environment.environment import Environment
from train import select_action
import torch
import sys
from torchvision.transforms import ToTensor
import time
from matplotlib import pyplot as plt

args = get_args()
class Game:
    def __init__(self, fps, env: Environment):
        pygame.init()
        self.fps = fps
        self.env = env
        self.width = self.env.grid_size[1] * args.cell_size
        self.height = self.env.grid_size[0] * args.cell_size
        self.screen = pygame.display.set_mode((self.width + 1, self.height + 2 * args.cell_size + 1))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font("assets/PressStart2P-Regular.ttf", 13)
        self.score = 0
        self.scale_food = False
        self.transform = ToTensor()
        self.episode = 0
        self.start = time.time()
        self.max_score = 0

    def play(self, agent=None, epsilon=0.0, epsilon_decay=0.0):
        running = True
        epsilon = epsilon
        scores = []
        rewards = []
        episodes = []

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5))
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

                    _, _, _, score = self.env.step(action, 0)
                    self.score += score
                    self.max_score = max(self.max_score, self.score)
                    self.draw("HUMAN")
            else:
                self.env.reset()
                self.score = 0
                total_reward = 0
                while not self.env.done:
                    with torch.no_grad():
                        state = torch.from_numpy(self.env.get_state())
                        action = select_action(state, agent, epsilon)
                        _, reward, _, score = self.env.step(action, 0)
                        self.score += score
                        total_reward += reward
                        self.max_score = max(self.max_score, self.score)
                        self.draw()
                self.episode += 1
                episodes.append(self.episode)
                scores.append(self.score)
                rewards.append(total_reward)

                ax1.clear()
                ax1.plot(episodes, scores, label="Score", color='blue')
                ax1.set_title("Score per Game")
                ax1.set_xlabel("Game")
                ax1.set_ylabel("Score")
                ax1.legend()
                # ax1.set_xticks(range(1, len(episodes) + 1))
                # ax1.set_yticks(range(0, int(max(scores)) + 5))

                ax2.clear()
                ax2.plot(episodes, rewards, label="Total Reward", color='orange')
                ax2.set_title("Reward per Game")
                ax2.set_xlabel("Game")
                ax2.set_ylabel("Reward")
                ax2.legend()
                # ax2.set_xticks(range(1, len(episodes) + 1))

                plt.tight_layout()
                plt.pause(0.01)
            if epsilon < 0.05:
                epsilon = 0
            else:
                epsilon = epsilon * epsilon_decay
        pygame.quit()
        sys.exit()


    def draw(self, player="MACHINE"):
        # Draw snake
        for element in self.env.snake.position:
            y, x = element * args.cell_size
            pygame.draw.rect(self.screen, (0, 255, 0), (x, y + 2 * args.cell_size, args.cell_size, args.cell_size))
            pygame.draw.rect(self.screen, (0, 113, 0), (x, y + 2 * args.cell_size, args.cell_size, args.cell_size), 10)

        # Draw food
        if not self.scale_food:
            y, x = self.env.food.position * args.cell_size
            pygame.draw.rect(self.screen, (255, 0, 0),(x, y + 2 * args.cell_size, args.cell_size, args.cell_size))

        # Draw text
        texts = [
            self.font.render(f"SCORE: {self.score:04d}", True, (255, 255, 255)),
            self.font.render(f"MAX SCORE: {self.max_score:04d}", True, (255, 255, 255)),
            self.font.render(f"PLAYER: {player}", True, (255, 255, 255)),
            self.font.render(f"GAME: {self.episode + 1:04d}", True, (255, 255, 255))
        ]
        spacing = 50
        total_width = sum(text.get_width() for text in texts) + spacing * (len(texts) - 1)
        start_x = (self.width - total_width) / 2
        for text in texts:
            self.screen.blit(text, (start_x, args.cell_size - text.get_height() / 2))
            start_x += text.get_width() + spacing

        # Draw line
        for x in range(0, self.width * args.cell_size + 1, args.cell_size):
            pygame.draw.line(self.screen, (200, 200, 200), (x, 2 * args.cell_size), (x, self.width * args.cell_size))
        for y in range(2 * args.cell_size, self.height * args.cell_size + 1, args.cell_size):
            pygame.draw.line(self.screen, (200, 200, 200), (0, y), (self.height * args.cell_size, y))

        pygame.display.flip()
        self.clock.tick(self.fps)
        self.screen.fill((0, 0, 0))

if __name__ == '__main__':
    agent = DeepQNetwork()
    agent.load_state_dict(torch.load('checkpoint/snake_dqn.pth'))
    game = Game(fps=0, env=Environment(args.grid_size))
    game.play(agent=agent, epsilon=0.0, epsilon_decay=0.95)