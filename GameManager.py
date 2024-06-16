import numpy as np
import random
import torch
import pygame


# 定数
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
BLOCK_SIZE = 30
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
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

COLORS = [
    (0, 0, 0),    # EMPTY
    (255, 0, 0),  # I
    (0, 255, 0),  # O
    (0, 0, 255),  # T
    (255, 255, 0),# S
    (0, 255, 255),# Z
    (255, 0, 255),# L
    (255, 165, 0) # J
]

#形とshapeのindexを持つクラス
class Shape:
    def __init__(self, shape, shape_type):
        self.shape = shape
        self.shape_type = shape_type

    def rotate(self):
        self.shape = np.rot90(self.shape)
        

class GameManager:
    def __init__(self):
        self.board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=int) # 20x10でゲーム盤を初期化
        self.current_shape = None
        self.current_position = [0, 0]
        self.next_shapes = [self.random_shape() for _ in range(3)]
        self.hold_shape = None
        self.hold_used = False
        self.score = 0
        self.spawn_new_shape()
        

    def random_shape(self):
        shape = random.choice(SHAPES)
        shape_type = SHAPES.index(shape) + 1
        return Shape(np.array(shape), shape_type)
    

    def spawn_new_shape(self):
        self.current_shape = self.next_shapes.pop(0)
        self.current_position = [0, BOARD_WIDTH // 2 - len(self.current_shape.shape[0]) // 2]
        self.next_shapes.append(self.random_shape()) # 次の形を追加
        self.hold_used = False
        if self.check_collision(self.current_shape.shape, self.current_position):
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
    

    #shapeを盤面に固定するメソッド
    def merge_shape_to_board(self):
        shape_height = len(self.current_shape.shape)
        shape_width = len(self.current_shape.shape[0])
        for row in range(shape_height):
            for col in range(shape_width):
                if self.current_shape.shape[row][col]:
                    self.board[self.current_position[0] + row][self.current_position[1] + col] = self.current_shape.shape_type
        self.clear_lines()
        
        self.score += 1
        

    def clear_lines(self):
        new_board = []
        lines_cleared = 0
        for row in self.board:
            if not all(row):
                new_board.append(row) # 行が埋まっていない場合は新しい盤面に追加
            else:
                lines_cleared += 1 # 行が埋まっている場合はカウント
        for _ in range(lines_cleared):
            new_board.insert(0, np.zeros(BOARD_WIDTH)) # 埋まった行数だけ新しい行を上部に追加
            
        self.board = np.array(new_board)
        if lines_cleared == 1:
            self.score += lines_cleared * 100 # スコアを更新
        elif lines_cleared == 2:
            self.score += lines_cleared * 300
        elif lines_cleared == 3:
            self.score += lines_cleared * 500
        elif lines_cleared == 4:
            self.score += lines_cleared * 800
        

    def game_over(self):
        print("Game Over")
        print("Score:", self.score)
        self.__init__()
        

    def move(self, direction):
        new_position = self.current_position.copy()
        if direction == "left":
            new_position = [self.current_position[0], self.current_position[1] - 1]
        elif direction == "right":
            new_position = [self.current_position[0], self.current_position[1] + 1]
        elif direction == "down":
            new_position = [self.current_position[0] + 1, self.current_position[1]]

        if direction == "rotate":
            self.rotate()
        elif direction == "hard_drop":
            self.hard_drop()
        else:
            if not self.check_collision(self.current_shape.shape, new_position):
                self.current_position = new_position
            elif direction == "down":
                self.merge_shape_to_board()
                self.score += 10
                self.spawn_new_shape()
            

    def rotate(self):
        self.current_shape.rotate()
        if self.check_collision(self.current_shape.shape, self.current_position):
            self.current_shape.rotate()
            

    def hold(self):
        if not self.hold_used:
            if self.hold_shape:
                self.current_shape, self.hold_shape = self.hold_shape, self.current_shape
            else:
                self.hold_shape = self.current_shape
                self.spawn_new_shape()
            self.current_position = [0, BOARD_WIDTH // 2 - len(self.current_shape.shape[0]) // 2]
            self.hold_used = True
            
    
    def hard_drop(self):
        while not self.check_collision(self.current_shape.shape, [self.current_position[0] + 1, self.current_position[1]]):
            self.current_position[0] += 1
        self.merge_shape_to_board()
        self.spawn_new_shape()
        
    
    def draw_board(screen, board, current_shape, current_shape_type, current_position, next_shapes, hold_shape, hold_shape_type, score):
        screen.fill(WHITE)

        # Draw the main game board
        for row in range(len(board)):
            for col in range(len(board[row])):
                pygame.draw.rect(screen, COLORS[int(board[row][col])], 
                                (col * BLOCK_SIZE, row * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), 0)

        # Draw the current shape
        shape_height = len(current_shape)
        shape_width = len(current_shape[0])
        for row in range(shape_height):
            for col in range(shape_width):
                if current_shape[row][col]:
                    pygame.draw.rect(screen, COLORS[current_shape_type], 
                                    ((current_position[1] + col) * BLOCK_SIZE, 
                                    (current_position[0] + row) * BLOCK_SIZE, 
                                    BLOCK_SIZE, BLOCK_SIZE), 0)

        # Draw the next shapes
        for i, shape_obj in enumerate(next_shapes):
            shape, shape_type = shape_obj.shape, shape_obj.shape_type
            for row in range(len(shape)):
                for col in range(len(shape[row])):
                    if shape[row][col]:
                        pygame.draw.rect(screen, COLORS[shape_type], 
                                        (SCREEN_WIDTH - 150 + col * BLOCK_SIZE, 50 + i * 100 + row * BLOCK_SIZE, 
                                        BLOCK_SIZE, BLOCK_SIZE), 0)

        # Draw the hold shape
        if hold_shape:
            shape = hold_shape.shape
            for row in range(len(shape)):
                for col in range(len(shape[row])):
                    if shape[row][col]:
                        pygame.draw.rect(screen, COLORS[hold_shape_type], 
                                        (SCREEN_WIDTH - 150 + col * BLOCK_SIZE, 400 + row * BLOCK_SIZE, 
                                        BLOCK_SIZE, BLOCK_SIZE), 0)

        # Draw the score
        font = pygame.font.Font(None, 36)
        score_text = font.render(f'Score: {score}', True, BLACK)
        screen.blit(score_text, (SCREEN_WIDTH - 150, 20))
            

    def get_board(self):
        return self.board

    def get_current_shape(self):
        return self.current_shape.shape

    def get_current_shape_type(self):
        return self.current_shape.shape_type

    def get_current_position(self):
        return self.current_position

    def get_next_shapes(self):
        return self.next_shapes

    def get_hold_shape(self):
        return self.hold_shape

    def get_hold_shape_type(self):
        return self.hold_shape.shape_type if self.hold_shape else None

    def get_score(self):
        return self.score

    def get_state(self):
        # ボードの状態をTensorに変換
        return torch.tensor(self.board, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    def is_game_over(self):
        return np.any(self.board[0, :])
    
    def reset(self):
        self.__init__()