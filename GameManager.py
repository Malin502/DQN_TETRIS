import numpy as np
import random

import pygame

# ボードサイズ
BOARD_WIDTH = 10
BOARD_HEIGHT = 20
# 定数
SCREEN_WIDTH = 300
SCREEN_HEIGHT = 600
BLOCK_SIZE = 30
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
COLORS = {
    0: (0, 0, 0),
    1: (255, 0, 0)
}

# テトリミノの形状
SHAPES = [
    np.array([[1, 1, 1, 1]]),          # I
    np.array([[1, 1], [1, 1]]),        # O
    np.array([[0, 1, 0], [1, 1, 1]]),  # T
    np.array([[0, 1, 1], [1, 1, 0]]),  # S
    np.array([[1, 1, 0], [0, 1, 1]]),  # Z
    np.array([[1, 0, 0], [1, 1, 1]]),  # L
    np.array([[0, 0, 1], [1, 1, 1]])   # J
]

# テトリミノの色
COLORS = [
    (0, 0, 0),       # 空
    (0, 255, 255),   # I
    (255, 255, 0),   # O
    (128, 0, 128),   # T
    (0, 255, 0),     # S
    (255, 0, 0),     # Z
    (255, 165, 0),   # L
    (0, 0, 255)      # J
]


class GameManager:
    def __init__(self):
        self.board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=int)
        self.current_shape = None
        self.current_shape_type = None
        self.hold_shape = None
        self.hold_shape_type = None
        self.next_shapes = [(random.choice(SHAPES), random.randint(1, len(SHAPES))) for _ in range(3)]
        self.current_position = [0, 0]
        self.reset()

    def reset(self):
        self.board.fill(0)
        self.new_shape()
        return self.get_state()

    def new_shape(self):
        self.current_shape, self.current_shape_type = self.next_shapes.pop(0)
        self.next_shapes.append((random.choice(SHAPES), random.randint(1, len(SHAPES))))
        self.current_position = [0, BOARD_WIDTH // 2 - self.current_shape.shape[1] // 2]
        self.rotation_count = 0  # 新しい形状が生成されるたびに回転回数をリセット

    def rotate(self):
        new_shape = np.rot90(self.current_shape)
        if not self.check_collision(new_shape, self.current_position):
            self.current_shape = new_shape
            self.rotation_count += 1

    def move(self, direction):
        new_position = self.current_position.copy()
        if direction == "left":
            new_position[1] -= 1
        elif direction == "right":
            new_position[1] += 1
        elif direction == "down":
            new_position[0] += 1


        if not self.check_collision(self.current_shape, new_position):
            self.current_position = new_position
            return True
        else:
            if direction == "down":
                self.lock_shape()
                self.clear_lines()
                self.new_shape()
            return False

    def hard_drop(self):
        while self.move("down"):
            pass

    def hold(self):
        if self.hold_shape is None:
            self.hold_shape = self.current_shape
            self.hold_shape_type = self.current_shape_type
            self.new_shape()
        else:
            self.hold_shape, self.current_shape = self.current_shape, self.hold_shape
            self.hold_shape_type, self.current_shape_type = self.current_shape_type, self.hold_shape_type
            self.current_position = [0, BOARD_WIDTH // 2 - self.current_shape.shape[1] // 2]
            self.rotation_count = 0

    def check_collision(self, shape, position):
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell and (
                        x + position[1] < 0 or
                        x + position[1] >= BOARD_WIDTH or
                        y + position[0] >= BOARD_HEIGHT or
                        self.board[y + position[0], x + position[1]]):
                    return True
        return False

    def lock_shape(self):
        for y, row in enumerate(self.current_shape):
            for x, cell in enumerate(row):
                if cell:
                    self.board[y + self.current_position[0], x + self.current_position[1]] = self.current_shape_type

    def clear_lines(self):
        lines_to_clear = [i for i, row in enumerate(self.board) if all(row)]
        for i in lines_to_clear:
            self.board[1:i + 1] = self.board[:i]
            self.board[0] = 0
        return len(lines_to_clear)

    def get_state(self):
        state = np.copy(self.board)
        for y, row in enumerate(self.current_shape):
            for x, cell in enumerate(row):
                if cell:
                    state[y + self.current_position[0], x + self.current_position[1]] = self.current_shape_type
        return state

    def game_over(self):
        return self.check_collision(self.current_shape, self.current_position)

    def get_score(self):
        return 0  # スコア計算のデフォルト値
    
    def shape_to_index(self, shape):
        for i, s in enumerate(SHAPES):
            if np.array_equal(shape, s):
                return i + 1
        return 0
    
    def draw_board(self, screen):
        screen.fill(WHITE)

        # Draw the main game board
        for y, row in enumerate(self.board):
            for x, cell in enumerate(row):
                color = COLORS[cell]
                pygame.draw.rect(screen, color, pygame.Rect(x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

        # Draw the current shape
        for y, row in enumerate(self.current_shape):
            for x, cell in enumerate(row):
                if cell:
                    color = COLORS[self.current_shape_type]
                    pygame.draw.rect(screen, color, pygame.Rect((self.current_position[1] + x) * BLOCK_SIZE, (self.current_position[0] + y) * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

        # Draw the hold shape
        if self.hold_shape is not None:
            for y, row in enumerate(self.hold_shape):
                for x, cell in enumerate(row):
                    if cell:
                        color = COLORS[self.hold_shape_type]
                        pygame.draw.rect(screen, color, pygame.Rect(11 * BLOCK_SIZE + x * BLOCK_SIZE, 1 * BLOCK_SIZE + y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

        # Draw the next shapes
        for i, (shape, shape_type) in enumerate(self.next_shapes):
            for y, row in enumerate(shape):
                for x, cell in enumerate(row):
                    if cell:
                        color = COLORS[shape_type]
                        pygame.draw.rect(screen, color, pygame.Rect(11 * BLOCK_SIZE + x * BLOCK_SIZE, (5 + 4 * i) * BLOCK_SIZE + y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
