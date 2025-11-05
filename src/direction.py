from enum import Enum

import numpy as np



class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    def __len__(self):
        return 4

    @staticmethod
    def sample():
        return Direction(np.random.randint(0, 3))

    def go_straight(self, position):
        if self is Direction.UP:
            return position + np.array([-1, 0])
        if self is Direction.RIGHT:
            return position + np.array([0, 1])
        if self is Direction.DOWN:
            return position + np.array([1, 0])
        return position + np.array([0, -1])