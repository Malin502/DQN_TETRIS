import numpy as np
import random

BOARD_WIDTH = 10
BOARD_HEIGHT = 20
EMPTY = 0

SHAPES = [
    [[1, 1, 1, 1]],  # I
    [[1, 1], [1, 1]],  # O
    [[1, 1, 1], [0, 1, 0]],  # T
    [[1, 1, 0], [0, 1, 1]],  # S
    [[0, 1, 1], [1, 1, 0]],  # Z
    [[1, 1, 1], [1, 0, 0]],  # L
    [[1, 1, 1], [0, 0, 1]],  # J
]

class GameManager:
    def __init__(self):
        self.board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=int)
        self.current_shape = None
        self.current_position = [0, 0]
        self.next_shapes = [random.choice(SHAPES) for _ in range(3)]
        self.hold_shape = None
        self.hold_used = False
        self.score = 0
        self.spawn_new_shape()

    def spawn_new_shape(self):
        if self.next_shapes:
            self.current_shape = self.next_shapes.pop(0)
        else:
            self.current_shape = random.choice(SHAPES)
        self.current_position = [0, BOARD_WIDTH // 2 - len(self.current_shape[0]) // 2]
        self.next_shapes.append(random.choice(SHAPES))
        self.hold_used = False
        if self.check_collision(self.current_shape, self.current_position):
            self.game_over()

    def check_collision(self, shape, position):
        shape_height = len(shape)
        shape_width = len(shape[0])
        for row in range(shape_height):
            for col in range(shape_width):
                if shape[row][col] and (row + position[0] >= BOARD_HEIGHT or
                                        col + position[1] < 0 or
                                        col + position[1] >= BOARD_WIDTH or
                                        self.board[row + position[0]][col + position[1]]):
                    return True
        return False

    def merge_shape_to_board(self):
        shape_height = len(self.current_shape)
        shape_width = len(self.current_shape[0])
        for row in range(shape_height):
            for col in range(shape_width):
                if self.current_shape[row][col]:
                    self.board[self.current_position[0] + row][self.current_position[1] + col] = 1
        self.clear_lines()

    def clear_lines(self):
        new_board = []
        lines_cleared = 0
        for row in self.board:
            if not all(row):
                new_board.append(row)
            else:
                lines_cleared += 1
        for _ in range(lines_cleared):
            new_board.insert(0, np.zeros(BOARD_WIDTH))
        self.board = np.array(new_board)
        self.score += lines_cleared * 100

    def game_over(self):
        print("Game Over")
        self.__init__()

    def move(self, direction):
        if direction == "left":
            new_position = [self.current_position[0], self.current_position[1] - 1]
        elif direction == "right":
            new_position = [self.current_position[0], self.current_position[1] + 1]
        elif direction == "down":
            new_position = [self.current_position[0] + 1, self.current_position[1]]

        if not self.check_collision(self.current_shape, new_position):
            self.current_position = new_position
        elif direction == "down":
            self.merge_shape_to_board()
            self.spawn_new_shape()

    def rotate(self):
        new_shape = np.rot90(self.current_shape)
        if not self.check_collision(new_shape, self.current_position):
            self.current_shape = new_shape.tolist()

    def hold(self):
        if not self.hold_used:
            if self.hold_shape:
                self.current_shape, self.hold_shape = self.hold_shape, self.current_shape
            else:
                self.hold_shape = self.current_shape
                self.spawn_new_shape()
            self.current_position = [0, BOARD_WIDTH // 2 - len(self.current_shape[0]) // 2]
            self.hold_used = True

    def get_board(self):
        return self.board

    def get_current_shape(self):
        return self.current_shape

    def get_current_position(self):
        return self.current_position

    def get_next_shapes(self):
        return self.next_shapes

    def get_hold_shape(self):
        return self.hold_shape

    def get_score(self):
        return self.score
