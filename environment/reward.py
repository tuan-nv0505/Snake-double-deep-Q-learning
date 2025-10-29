from collections import deque
from utils.direction import Direction
from utils.utils import is_collision, is_position_valid, position_neighbor
import numpy as np


class Reward:
    def __init__(self, env):
        self.env = env

    def eaten(self, action, reward_value):
        if is_collision(self.__snake_by_action(action)[0], self.env.food.position):
            return reward_value
        return -0.5

    def dead(self, action, reward_value):
        snake_by_action = self.__snake_by_action(action)
        head, body = snake_by_action[0], snake_by_action[1:]
        if is_collision(head, body) or not is_position_valid(head, self.env.grid_size):
            return -reward_value
        return 0

    def reward_by_distance_delta(self, action, reward_value):
        snake_by_action = self.__snake_by_action(action)
        old_head = self.env.snake.position[0]
        new_head = snake_by_action[0]
        old_distance = abs(old_head[0] - self.env.food.position[0]) + abs(old_head[1] - self.env.food.position[1])
        new_distance = abs(new_head[0] - self.env.food.position[0]) + abs(new_head[1] - self.env.food.position[1])
        if new_distance < old_distance:
            return reward_value
        return -reward_value

    def avoiding_imminent_danger(self, action, reward_value):
        snake_straight = self.__snake_by_action(1)  # 0:left, 1:straight, 2:right
        snake_by_action = self.__snake_by_action(action)

        danger_ahead = (
            is_collision(snake_straight[0], snake_straight[1:])
            or not is_position_valid(snake_straight[0], self.env.grid_size)
        )
        safe_action = not (
            is_collision(snake_by_action[0], snake_by_action[1:])
            or not is_position_valid(snake_by_action[0], self.env.grid_size)
        )

        if danger_ahead and safe_action:
            return reward_value
        return 0

    def moving_same_direction(self, action, reward_value):
        if action == 1:
            return reward_value
        return -reward_value

    def move_not_safe(self, action, reward_value):
        snake_by_action = self.__snake_by_action(action)
        head = snake_by_action[0]

        queue = deque()
        visited = np.zeros(self.env.grid_size)
        if not is_position_valid(head, self.env.grid_size) or is_collision(head, snake_by_action[1:]):
            return 0
        row, col = head
        queue.append(head)
        visited[row, col] = 1

        spaces_safe = 0
        while queue:
            pos_current = queue.popleft()
            for next_pos in position_neighbor(pos_current):
                if not is_position_valid(next_pos, self.env.grid_size) or is_collision(next_pos, snake_by_action[1:]):
                    continue
                r, c = next_pos
                if visited[r, c] == 0:
                    visited[r, c] = 1
                    spaces_safe += 1
                    queue.append(next_pos)

                    if spaces_safe > len(self.env.snake.position):
                        return 0
        return reward_value

    def __call__(self, action, epsilon):
        rw = 0
        rw += self.eaten(action, 100)
        rw += self.dead(action, -(100 + 2.5 / (epsilon + 1e-4)))
        rw += self.reward_by_distance_delta(action, 2)

        if a > 0:
            print(a, b, c, ' : ', a + b +c)

        if epsilon <= 0.3:
            rw += self.avoiding_imminent_danger(action, 3)
            rw += self.move_not_safe(action, -(50 + 2.5 / (epsilon + 1e-4)))

        if epsilon <= 0.2:
            rw += self.moving_same_direction(action, 1)
        return rw

    def __snake_by_action(self, action):
        new_direction = Direction((self.env.snake.direction.value + action - 1 + len(Direction)) % len(Direction))
        new_head = new_direction.go_straight(self.env.snake.head)
        new_body = np.vstack((np.array(new_head), self.env.snake.position))
        if not is_collision(new_head, self.env.food.position):
            new_body = new_body[:-1, :]
        return new_body
