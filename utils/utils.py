import numpy as np

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