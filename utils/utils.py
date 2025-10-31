import numpy as np
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=10000)
    parser.add_argument('--epsilon_start', type=float, default=1.0)
    parser.add_argument('--epsilon_end', type=float, default=0.05)
    parser.add_argument('--epsilon_decay', type=float, default=0.9995)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--memory_replay', type=int, default=50000)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--grid_size', type=lambda x: tuple(map(int, x.split(','))), default=(30, 46))
    parser.add_argument('--cell_size', type=int, default=30)
    parser.add_argument('--logging', type=str, default='tensorboard')
    parser.add_argument('--tau', type=float, default=0.005)

    args = parser.parse_args()
    return args

def is_collision(position_1, position_2):
    position_1 = np.array(position_1)
    position_2 = np.array(position_2)
    if position_1.ndim == 1 and position_2.ndim == 1:
        return np.array_equal(position_1, position_2)
    return np.any(np.all(position_1 == position_2, axis=1))

def is_position_valid(position, grid_size):
    return 0 <= position[0] < grid_size[0] and 0 <= position[1] < grid_size[1]

def position_neighbor(position):
    return np.array([
        position + [-1, 0],
        position + [0, 1],
        position + [1, 0],
        position + [0, -1]
    ])
