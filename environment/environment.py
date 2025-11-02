from pathlib import Path
import sys
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import numpy as np
from utils.direction import Direction
from environment.reward import Reward
from utils.utils import get_args, is_collision, is_position_valid, position_neighbor

args = get_args()
class Environment:
    def __init__(self, grid_size):
        super().__init__()
        self.grid_size = grid_size
        self.snake = Snake(grid_size)
        self.food = Food(grid_size)
        self.food.reset_position(invalid_position=self.snake.position)
        self.done = False

    def step(self, action, epsilon):
        reward = Reward(env=self)
        reward_value = reward(action, epsilon)
        self.snake.update_direction(action)
        score = self.snake.move(self.food)
        if not self.snake.is_alive():
            self.done = True
        return self.get_state(), reward_value, self.done, score

    def reset(self):
        self.snake.reset()
        self.food.reset_position(invalid_position=self.snake.position)
        self.done = False


    def get_state(self):
        pos_head = self.snake.head
        pos_food = self.food.position
        neighbors_head = position_neighbor(pos_head)
        neighbors_head = neighbors_head[np.arange(4) != (self.snake.direction.value - 2 + 4) % 4]

        return np.array([
            pos_head[0] / self.grid_size[0],
            pos_head[1] / self.grid_size[1],
            pos_food[0] / self.grid_size[0],
            pos_food[1] / self.grid_size[1],
            self.snake.direction.value,
            self.snake.position.shape[0],
            int(is_position_valid(neighbors_head[0], self.grid_size) and not is_collision(neighbors_head[0], self.snake.position)),
            int(is_position_valid(neighbors_head[1], self.grid_size) and not is_collision(neighbors_head[1], self.snake.position)),
            int(is_position_valid(neighbors_head[2], self.grid_size) and not is_collision(neighbors_head[2], self.snake.position))
        ], dtype=np.float32)


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
