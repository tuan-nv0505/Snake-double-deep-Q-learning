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


class Environment(gym.Env):
    def __init__(self, grid_size):
        super().__init__()
        self.grid_size = grid_size
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=255, shape=(grid_size[0], grid_size[1], 3), dtype=np.uint8)
        self.snake = Snake(grid_size)
        self.food = Food(grid_size)
        self.food.reset_position(invalid_position=self.snake.position)
        self.obs = self.__get_obs()
        self.done = False

    def step(self, action):
        reward = Reward(env=self)
        reward_value = reward.reward_value(action)
        self.snake.update_direction(action)
        self.snake.move(self.food)
        if not self.snake.is_alive():
            self.done = True
        self.obs = self.__get_obs()
        return self.obs, reward_value, self.done

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.snake.reset()
        self.food.reset_position(invalid_position=self.snake.position)
        self.done = False
        return self.__get_obs()

    def render(self):
        print(self.obs[:, :, 0] + self.obs[:, :, 1] + self.obs[:, :, 2])


    def __get_obs(self):
        obs = np.zeros((*self.grid_size, 3), dtype=np.uint8)
        if self.snake.is_alive():
            obs[self.snake.head[0], self.snake.head[1], 0] = 255
        for x, y in self.snake.position[1:]:
            obs[x, y, 1] = 255
        obs[self.food.position[0], self.food.position[1], 2] = 255
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
        else:
            self.position = self.position[:-1, :]

    def update_direction(self, action):
        self.direction = Direction((self.direction.value + action - 1 + len(Direction)) % len(Direction))

    def reset(self):
        self.head = np.array([self.grid_size[0] // 2, self.grid_size[1] // 2])
        self.position = np.array([
            self.head,
            (self.head[0], self.head[1] - 1),
            (self.head[0], self.head[1] - 2),
        ])

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

