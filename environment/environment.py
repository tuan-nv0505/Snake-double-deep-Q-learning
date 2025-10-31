from pathlib import Path
import sys
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from utils.direction import Direction
from utils.utils import is_position_valid, is_collision
from environment.reward import Reward
from utils.utils import get_args

args = get_args()
class Environment(gym.Env):
    def __init__(self, grid_size, epsilon):
        super().__init__()
        self.grid_size = grid_size
        self.epsilon = epsilon
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=255, shape=args.grid_size, dtype=np.float32)
        self.snake = Snake(grid_size)
        self.food = Food(grid_size)
        self.food.reset_position(invalid_position=self.snake.position)
        self.obs = self.get_obs()
        self.done = False

    def step(self, action):
        reward = Reward(env=self)
        reward_value = reward(action, self.epsilon)
        self.snake.update_direction(action)
        score = self.snake.move(self.food)
        if not self.snake.is_alive():
            self.done = True
        self.obs = self.get_obs()
        return self.obs, reward_value, self.done, score

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.snake.reset()
        self.food.reset_position(invalid_position=self.snake.position)
        self.done = False
        return self.get_obs()

    def render(self):
        print(self.obs)

    def get_obs(self):
        obs = np.zeros(self.grid_size, dtype=np.float32)
        snake_position = self.snake.position
        try:
            obs[snake_position[0][0], snake_position[0][1]]
        except IndexError:
            obs[snake_position[1:,0], snake_position[1:,1]] = 1
        else:
            obs[snake_position[0][0], snake_position[0][1]] = 2
            obs[snake_position[:, 0], snake_position[:, 1]] = 1
        obs[self.food.position[0], self.food.position[1]] = -1
        return obs

class Snake:
    def __init__(self, grid_size, position=None):
        self.grid_size = grid_size
        self.head = np.array([grid_size[0] // 2, grid_size[1] // 2])
        self.position = np.array([
            self.head,
            (self.head[0], self.head[1] - 1),
            (self.head[0], self.head[1] - 2),
        ])
        if position is not None:
            self.position = position
            self.head = self.position[0]
        self.direction = Direction.RIGHT

    def is_alive(self):
        return is_position_valid(self.head, self.grid_size) and not(is_collision(self.head, self.position[1:]))

    def move(self, food):
        new_head = self.direction.go_straight(self.head)
        self.position = np.vstack((np.array(new_head), self.position))
        self.head = self.position[0]
        if is_collision(self.head, food.position):
            food.reset_position(invalid_position=self.position)
            return 1
        self.position = self.position[:-1, :]
        return 0

    def update_direction(self, action):
        self.direction = Direction((self.direction.value + action - 1 + len(Direction)) % len(Direction))

    def reset(self):
        self.head = np.array([self.grid_size[0] // 2, self.grid_size[1] // 2])
        self.position = np.array([
            self.head,
            (self.head[0], self.head[1] - 1),
            (self.head[0], self.head[1] - 2),
        ])
        self.direction = Direction.RIGHT

class Food:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.position = None

    def reset_position(self, invalid_position=None):
        self.position = self.__create_position()
        if invalid_position is not None:
            while True:
                if invalid_position.ndim == 1:
                    if np.all(self.position == invalid_position):
                        self.position = self.__create_position()
                    else:
                        break
                else:
                    if np.any(np.all(self.position == invalid_position, axis=1)):
                        self.position = self.__create_position()
                    else:
                        break

    def __create_position(self):
        return np.array([
            np.random.randint(self.grid_size[0]),
            np.random.randint(self.grid_size[1])
        ])

# python -m environment.environment --grid_size 10,10
if __name__ == '__main__':
    # test environment with action: 0(left) 1(straight) 2(right)
    env = Environment(args.grid_size, 1)
    while True:
        env.reset()
        while not env.done:
            action = int(input())
            env.step(action)
            env.render()
