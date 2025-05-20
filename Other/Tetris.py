import pygame
import numpy as np
import torch
import random

from Agent import DeepQNetwork

# 初期化
pygame.init()
pygame.font.init()

# 定数
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
PURPLE = (147, 88, 254)
GREEN = (54, 175, 144)
CYAN = (102, 217, 238)
ORANGE = (254, 151, 32)
BLUE = (0, 0, 255)

PIECE_COLORS = [BLACK, YELLOW, PURPLE, GREEN, RED, CYAN, ORANGE, BLUE]

PIECES = [
    [[1, 1],
     [1, 1]],

    [[0, 2, 0],
     [2, 2, 2]],

    [[0, 3, 3],
     [3, 3, 0]],

    [[4, 4, 0],
     [0, 4, 4]],

    [[5, 5, 5, 5]],

    [[0, 0, 6],
     [6, 6, 6]],

    [[7, 0, 0],
     [7, 7, 7]]
]

class Tetris:
    def __init__(self, height=20, width=10, block_size=30):
        self.height = height
        self.width = width
        self.block_size = block_size
        self.reset()
        self.screen = pygame.display.set_mode((self.width * self.block_size + 200, self.height * self.block_size))  # ネクストミノ表示のために画面幅を拡張
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 25)
        self.latest_y_pos = 0
        
    def reset(self):
        self.board = [[0] * self.width for _ in range(self.height)]
        self.score = 0
        self.tetrominoes = 0
        self.cleared_lines = 0
        self.bag = list(range(len(PIECES)))
        random.shuffle(self.bag)
        self.ind = self.bag.pop()
        self.piece = [row[:] for row in PIECES[self.ind]]
        self.current_pos = {"x": self.width // 2 - len(self.piece[0]) // 2, "y": 0}
        self.next_pieces = self.get_next_pieces(3)  # ネクストミノ3つを生成
        self.gameover = False
        return self.get_state_properties(self.board)
    
    def get_next_pieces(self, num):
        next_pieces = []
        for _ in range(num):
            if not len(self.bag):
                self.bag = list(range(len(PIECES)))
                random.shuffle(self.bag)
            next_pieces.append(self.bag.pop())
        return next_pieces
    
    def new_piece(self):
        self.ind = self.next_pieces.pop(0)
        self.piece = [row[:] for row in PIECES[self.ind]]
        self.current_pos = {"x": self.width // 2 - len(self.piece[0]) // 2, "y": 0}
        if not len(self.bag):
            self.bag = list(range(len(PIECES)))
            random.shuffle(self.bag)
        self.next_pieces.append(self.bag.pop())
        if self.check_collision(self.piece, self.current_pos):
            self.gameover = True

    def get_state(self):
        return self.ind, self.next_pieces
    
    def rotate(self, piece):
        return [list(row)[::-1] for row in zip(*piece)]

    def get_state_properties(self, board):
        lines_cleared, board = self.check_cleared_rows(board)
        holes = self.get_holes(board)
        bumpiness, height = self.get_bumpiness_and_height(board)
        ind, next_pieces = self.get_state()
        above_block_squared_sum = self.get_above_block_squared_sum(board)
        center_max_height = self.get_center_max_height(board)
        row_transitions = self.get_row_transitions(board)
        column_transitions = self.get_column_transitions(board)

        return torch.FloatTensor([
            lines_cleared, 
            holes, 
            bumpiness, 
            height, 
            above_block_squared_sum, 
            #center_max_height,
            #row_transitions,
            #column_transitions,
            ind, 
            next_pieces[0], next_pieces[1], next_pieces[2]
        ])
        
    def get_holes(self, board):
        num_holes = 0
        for col in zip(*board):
            row = 0
            while row < self.height and col[row] == 0:
                row += 1
            num_holes += len([x for x in col[row + 1:] if x == 0])
        return num_holes

    def get_bumpiness_and_height(self, board):
        board = np.array(board)
        mask = board != 0
        invert_heights = np.where(mask.any(axis=0), np.argmax(mask, axis=0), self.height)
        heights = self.height - invert_heights
        total_height = np.sum(heights)
        currs = heights[:-1]
        nexts = heights[1:]
        diffs = np.abs(currs - nexts)
        total_bumpiness = np.sum(diffs)
        return total_bumpiness, total_height
    
    def get_above_block_squared_sum(self, board):
        above_block_squared_sum = 0
        for x in range(self.width):
            block_count = 0
            for y in range(self.height):
                if board[y][x] != 0:
                    block_count += 1
                else:
                    above_block_squared_sum += block_count ** 2
        return above_block_squared_sum
    
    def get_center_max_height(self, board):
        center_columns = [board[y][self.width // 2 - 1:self.width // 2 + 1] for y in range(self.height)]
        max_height = 0
        for col in range(len(center_columns[0])):
            height = 0
            for row in range(len(center_columns)):
                if center_columns[row][col] != 0:
                    height = self.height - row
                    break
            if height > max_height:
                max_height = height
        return max_height

    def get_row_transitions(self, board):
        row_transitions = 0
        for y in range(self.height):
            for x in range(self.width - 1):
                if (board[y][x] == 0 and board[y][x + 1] != 0) or (board[y][x] != 0 and board[y][x + 1] == 0):
                    row_transitions += 1
        return row_transitions

    def get_column_transitions(self, board):
        column_transitions = 0
        for x in range(self.width):
            for y in range(self.height - 1):
                if (board[y][x] == 0 and board[y + 1][x] != 0) or (board[y][x] != 0 and board[y + 1][x] == 0):
                    column_transitions += 1
        return column_transitions

    def get_next_states(self):
        states = {}
        lines_cleared_dict = {}
        piece_id = self.ind
        curr_piece = [row[:] for row in self.piece]

        # ピースによって回転数の設定
        if piece_id == 0:  # O piece
            num_rotations = 1
        elif piece_id == 2 or piece_id == 3 or piece_id == 4:
            num_rotations = 2
        else:
            num_rotations = 4

        for i in range(num_rotations):
            valid_xs = self.width - len(curr_piece[0])
            for x in range(valid_xs + 1):
                piece = [row[:] for row in curr_piece]
                pos = {"x": x, "y": 0}
                while not self.check_collision(piece, pos):
                    pos["y"] += 1
                self.truncate(piece, pos)
                board = self.store(piece, pos)
                state_properties = self.get_state_properties(board)
                lines_cleared, _ = self.check_cleared_rows(board)
                states[(x, i)] = state_properties
                lines_cleared_dict[(x, i)] = lines_cleared
            curr_piece = self.rotate(curr_piece)
        return states, lines_cleared_dict



    def get_current_board_state(self):
        board = [x[:] for x in self.board]
        for y in range(len(self.piece)):
            for x in range(len(self.piece[y])):
                board[y + self.current_pos["y"]][x + self.current_pos["x"]] = self.piece[y][x]
        return board
    
    def get_line_score(self, cleared_lines):
        if cleared_lines == 1:
            return 20
        elif cleared_lines == 2:
            return 50
        elif cleared_lines == 3:
            return 90
        elif cleared_lines == 4:
            return 140
        else:
            return 0


    def check_collision(self, piece, pos):
        future_y = pos["y"] + 1
        for y in range(len(piece)):
            for x in range(len(piece[y])):
                if future_y + y > self.height - 1 or self.board[future_y + y][pos["x"] + x] and piece[y][x]:
                    return True
        return False

    def truncate(self, piece, pos):
        gameover = False
        last_collision_row = -1
        for y in range(len(piece)):
            for x in range(len(piece[y])):
                if self.board[pos["y"] + y][pos["x"] + x] and piece[y][x]:
                    if y > last_collision_row:
                        last_collision_row = y

        if pos["y"] - (len(piece) - last_collision_row) < 0 and last_collision_row > -1:
            while last_collision_row >= 0 and len(piece) > 1:
                gameover = True
                last_collision_row = -1
                del piece[0]
                for y in range(len(piece)):
                    for x in range(len(piece[y])):
                        if self.board[pos["y"] + y][pos["x"] + x] and piece[y][x] and y > last_collision_row:
                            last_collision_row = y
        return gameover

    def store(self, piece, pos):
        board = [x[:] for x in self.board]
        for y in range(len(piece)):
            for x in range(len(piece[y])):
                if piece[y][x] and not board[y + pos["y"]][x + pos["x"]]:
                    board[y + pos["y"]][x + pos["x"]] = piece[y][x]
        return board

    def check_cleared_rows(self, board):
        to_delete = []
        for i, row in enumerate(board[::-1]):
            if 0 not in row:
                to_delete.append(len(board) - 1 - i)
        if len(to_delete) > 0:
            board = self.remove_row(board, to_delete)
        return len(to_delete), board

    def remove_row(self, board, indices):
        for i in indices[::-1]:
            del board[i]
            board = [[0 for _ in range(self.width)]] + board
        return board

    def step(self, action):
        x, num_rotations = action
        self.current_pos = {"x": x, "y": 0}
        for _ in range(num_rotations):
            self.piece = self.rotate(self.piece)

        while not self.check_collision(self.piece, self.current_pos):
            self.current_pos["y"] += 1
            self.render()
        self

        overflow = self.truncate(self.piece, self.current_pos)
        if overflow:
            self.gameover = True

        self.board = self.store(self.piece, self.current_pos)
        self.latest_y_pos = self.current_pos["y"]

        lines_cleared, self.board = self.check_cleared_rows(self.board)
        score = 1 + self.get_line_score(lines_cleared)
        self.score += score
        self.tetrominoes += 1
        self.cleared_lines += lines_cleared
        if not self.gameover:
            self.new_piece()
        if self.gameover:
            self.score -= 3

        return score, self.gameover

    def render(self):
        self.screen.fill(BLACK)
        for y in range(self.height):
            for x in range(self.width):
                pygame.draw.rect(self.screen, PIECE_COLORS[self.board[y][x]], 
                                 (x * self.block_size, y * self.block_size, self.block_size, self.block_size))
                pygame.draw.rect(self.screen, WHITE, 
                                 (x * self.block_size, y * self.block_size, self.block_size, self.block_size), 1)
        
        for y in range(len(self.piece)):
            for x in range(len(self.piece[y])):
                if self.piece[y][x]:
                    pygame.draw.rect(self.screen, PIECE_COLORS[self.piece[y][x]], 
                                     ((self.current_pos["x"] + x) * self.block_size, 
                                      (self.current_pos["y"] + y) * self.block_size, 
                                      self.block_size, self.block_size))
                    pygame.draw.rect(self.screen, WHITE, 
                                     ((self.current_pos["x"] + x) * self.block_size, 
                                      (self.current_pos["y"] + y) * self.block_size, 
                                      self.block_size, self.block_size), 1)
        
        # ネクストミノの描画
        next_piece_offset = self.width * self.block_size + 30
        for i, next_piece_index in enumerate(self.next_pieces):
            piece = PIECES[next_piece_index]
            for y in range(len(piece)):
                for x in range(len(piece[y])):
                    if piece[y][x]:
                        pygame.draw.rect(self.screen, PIECE_COLORS[piece[y][x]], 
                                         (next_piece_offset + x * self.block_size, 50 + i * 100 + y * self.block_size, self.block_size, self.block_size))
                        pygame.draw.rect(self.screen, WHITE, 
                                         (next_piece_offset + x * self.block_size, 50 + i * 100 + y * self.block_size, self.block_size, self.block_size), 1)
        
        score_text = self.font.render(f'Score: {self.score}', True, WHITE)
        self.screen.blit(score_text, (self.width * self.block_size + 70, self.height * self.block_size - 80))
        pygame.display.flip()
        
    def get_cleared_lines(self):
        return self.cleared_lines
    
    
    
    
# デバッグ用のメイン
def main():
    game = Tetris()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    game.current_pos["x"] -= 1
                    if game.check_collision(game.piece, game.current_pos):
                        game.current_pos["x"] += 1
                elif event.key == pygame.K_RIGHT:
                    game.current_pos["x"] += 1
                    if game.check_collision(game.piece, game.current_pos):
                        game.current_pos["x"] -= 1
                elif event.key == pygame.K_DOWN:
                    game.current_pos["y"] += 1
                    if game.check_collision(game.piece, game.current_pos):
                        game.current_pos["y"] -= 1
                elif event.key == pygame.K_UP:
                    game.piece = game.rotate(game.piece)
                    if game.check_collision(game.piece, game.current_pos):
                        for _ in range(3):
                            game.piece = game.rotate(game.piece)
                elif event.key == pygame.K_SPACE:
                    while not game.check_collision(game.piece, game.current_pos):
                        game.current_pos["y"] += 1
                    game.current_pos["y"] -= 1
                    game.render()
                    pygame.display.update()
                    while not game.check_collision(game.piece, game.current_pos):
                        game.current_pos["y"] += 1
                        game.render()

                    overflow = game.truncate(game.piece, game.current_pos)
                    if overflow:
                        game.gameover = True

                    game.board = game.store(game.piece, game.current_pos)
                    game.latest_y_pos = game.current_pos["y"]

                    lines_cleared, game.board = game.check_cleared_rows(game.board)
                    score = 1 + game.get_line_score(lines_cleared)
                    game.score += score
                    game.tetrominoes += 1
                    game.cleared_lines += lines_cleared
                    if not game.gameover:
                        game.new_piece()
                    if game.gameover:
                        game.score -= 3
                        
                elif event.key == pygame.K_r:
                    state = game.get_state_properties(game.board)
                    print(state)
                game.render()
                pygame.display.update()
        game.clock.tick(5)

    pygame.quit()

if __name__ == "__main__":
    main()
