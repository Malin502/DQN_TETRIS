import numpy as np
import random
import pygame

# ボードサイズ
BOARD_WIDTH = 10
BOARD_HEIGHT = 20
# 定数
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
BUTTON_WIDTH = 100
BUTTON_HEIGHT = 50
BLOCK_SIZE = 30
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
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




class Shape:
    def __init__(self, shape_index):
        self.shape = SHAPES[shape_index]
        self.color = COLORS[shape_index + 1]
        self.type = shape_index + 1


class GameManager:
    def __init__(self):
        self.board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=int)
        self.current_shape = None
        self.hold_shape = None
        self.next_shapes = [Shape(random.randint(0, len(SHAPES) - 1)) for _ in range(3)]
        self.current_position = [0, 0]
        self.score = 0
        self.latest_clear_mino_height = 0
        self.latest_y_position = 0
        self.reset()

    def reset(self):
        self.board.fill(0)
        self.new_shape()
        self.score = 0
        self.latest_clear_mino_height = 0
        return self.get_state()

    def new_shape(self):
        self.current_shape = self.next_shapes.pop(0)
        self.next_shapes.append(Shape(random.randint(0, len(SHAPES) - 1)))
        self.current_position = [0, BOARD_WIDTH // 2 - self.current_shape.shape.shape[1] // 2]  # 初期位置を中央に設定
        self.rotation_count = 0  # 新しい形状が生成されるたびに回転回数をリセット

    def rotate(self):
        new_shape = np.rot90(self.current_shape.shape)
        if not self.check_collision(new_shape, self.current_position):
            self.current_shape.shape = new_shape
            self.rotation_count += 1

    def move(self, direction):
        new_position = self.current_position.copy()
        if direction == "left":
            new_position[1] -= 1
        elif direction == "right":
            new_position[1] += 1
        elif direction == "down":
            new_position[0] += 1
        elif direction == "rotate":
            self.rotate()
        elif direction == "hard_drop":
            self.hard_drop()
            return True

        if not self.check_collision(self.current_shape.shape, new_position):
            self.current_position = new_position
            return True
        else:
            if direction == "down":
                self.lock_shape()
                lines_cleared = self.clear_lines()
                self.latest_y_position = self.current_position[0]
                self.score += 1 * self.latest_y_position / 10 + self.calculate_line_clear_score(lines_cleared)
                self.score = round(self.score, 2)
                self.new_shape()
            return False


    def action(self, action):
        if(action[2] == 1):
            self.hold()
        else:
            for _ in range(action[1]):
                self.rotate()
            self.current_position[1] += action[0]
            
            if not self.check_collision(self.current_shape.shape, self.current_position):
                self.hard_drop()

        return self.latest_y_position
    
    
    def hard_drop(self):
        while self.move("down"):
            pass
        

    def hold(self):
        if self.current_shape is None:
            return

        if self.hold_shape is None:
            self.hold_shape = self.current_shape
            self.new_shape()
        else:
            # スワップ
            self.hold_shape, self.current_shape = self.current_shape, self.hold_shape
            self.current_position = [0, BOARD_WIDTH // 2]
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
        for y, row in enumerate(self.current_shape.shape):
            for x, cell in enumerate(row):
                if cell:
                    self.board[y + self.current_position[0], x + self.current_position[1]] = self.current_shape.type

    def clear_lines(self):
        lines_to_clear = [i for i, row in enumerate(self.board) if all(row)]
        for i in lines_to_clear:
            self.latest_clear_mino_height = BOARD_HEIGHT - i
            self.board[1:i + 1] = self.board[:i]
            self.board[0] = 0
        return len(lines_to_clear)
    
    #スコアに応じて消えたライン数を返す
    def lines_cleard(self, score):
        if score >= 800:
            return 4
        elif score >= 500:
            return 3
        elif score >= 300:
            return 2
        elif score >= 100:
            return 1
        else:
            return 0
    
    
    def calculate_line_clear_score(self, lines_cleared):
        if lines_cleared == 1:
            return 100
        elif lines_cleared == 2:
            return 300
        elif lines_cleared == 3:
            return 500
        elif lines_cleared == 4:
            return 800
        return 0

    def get_state(self):
        state = np.copy(self.board)
        for y, row in enumerate(self.current_shape.shape):
            for x, cell in enumerate(row):
                if cell:
                    state[y + self.current_position[0], x + self.current_position[1]] = self.current_shape.type
        return state
    
    
    def game_over(self):
        return self.check_collision(self.current_shape.shape, self.current_position)
    
    
    def get_score(self):    
        return self.score
    
    
    def draw_board(self, screen):
        screen.fill(WHITE)

        # Draw the main game board
        for y, row in enumerate(self.board):
            for x, cell in enumerate(row):
                color = COLORS[cell]
                pygame.draw.rect(screen, color, pygame.Rect(x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

        # Draw the current shape
        for y, row in enumerate(self.current_shape.shape):
            for x, cell in enumerate(row):
                if cell:
                    color = self.current_shape.color
                    pygame.draw.rect(screen, color, pygame.Rect((self.current_position[1] + x) * BLOCK_SIZE, (self.current_position[0] + y) * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

        # Draw the hold shape
        if self.hold_shape is not None:
            for y, row in enumerate(self.hold_shape.shape):
                for x, cell in enumerate(row):
                    if cell:
                        color = self.hold_shape.color
                        pygame.draw.rect(screen, color, pygame.Rect(11 * BLOCK_SIZE + x * BLOCK_SIZE, 1 * BLOCK_SIZE + y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

        # Draw the next shapes
        for i, shape in enumerate(self.next_shapes):
            for y, row in enumerate(shape.shape):
                for x, cell in enumerate(row):
                    if cell:
                        color = shape.color
                        pygame.draw.rect(screen, color, pygame.Rect(11 * BLOCK_SIZE + x * BLOCK_SIZE, (8 + 4 * i) * BLOCK_SIZE + y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

         # Draw the labels
        font = pygame.font.Font(None, 36)
        hold_text = font.render('hold', True, BLACK)
        next_text = font.render('next', True, BLACK)
        score_text = font.render(f'Score: {self.score}', True, BLACK)

        screen.blit(hold_text, (11 * BLOCK_SIZE, 0))
        screen.blit(next_text, (11 * BLOCK_SIZE, 5 * BLOCK_SIZE))
        screen.blit(score_text, (SCREEN_WIDTH - score_text.get_width() - 10, 10))
        
        # Draw the save button
        pygame.draw.rect(screen, BLACK, (SCREEN_WIDTH - BUTTON_WIDTH - 10, SCREEN_HEIGHT - BUTTON_HEIGHT - 10, BUTTON_WIDTH, BUTTON_HEIGHT))
        save_text = font.render('Save', True, WHITE)
        screen.blit(save_text, (SCREEN_WIDTH - BUTTON_WIDTH - 10 + 25, SCREEN_HEIGHT - BUTTON_HEIGHT - 10 + 10))
        
        
    def is_button_clicked(self, pos, button_rect):
        return button_rect.collidepoint(pos)
    
    
    #現在のボードの状態から次のボードの状態をシミュレートする
    def simulate_next_boards(self):
        simulated_boards = []
        actions = []
        scores = []
        # シミュレーション用のゲームマネージャーインスタンスを作成
        simulation_manager = GameManager()
        simulation_manager.board = np.copy(self.board)
        simulation_manager.current_shape = Shape(self.current_shape.type - 1)
        simulation_manager.current_shape.shape = np.copy(self.current_shape.shape)
        simulation_manager.hold_shape = Shape(self.hold_shape.type - 1) if self.hold_shape is not None else None
        if self.hold_shape is not None:
            simulation_manager.hold_shape.shape = np.copy(self.hold_shape.shape)
        simulation_manager.next_shapes = [Shape(shape.type - 1) for shape in self.next_shapes]
        simulation_manager.current_position = self.current_position.copy()
        simulation_manager.score = 0
        simulation_manager.latest_clear_mino_height = self.latest_clear_mino_height

        # ホールドしていない場合の状態シミュレート
        for dx in range(-5, 7):
            for rotation in range(4):
                simulation_manager.current_position = self.current_position.copy()
                simulation_manager.current_shape.shape = np.copy(self.current_shape.shape)
                for _ in range(rotation):
                    simulation_manager.rotate()
                simulation_manager.current_position[1] += dx
                if not simulation_manager.check_collision(simulation_manager.current_shape.shape, simulation_manager.current_position):
                    simulation_manager.hard_drop()
                    actions.append([dx, rotation, 0])
                    scores.append(simulation_manager.score)
                    simulated_boards.append(np.copy(simulation_manager.board))
                simulation_manager.board = np.copy(self.board)
                
                simulation_manager.score = 0

        # ホールド時の状態シミュレート
        if self.current_shape is not None:
            # 現在の状態を一時保存
            actions.append([0, 0, 1])
            scores.append(0)
            
            temp_current_shape = simulation_manager.current_shape
            temp_hold_shape = simulation_manager.hold_shape
            temp_current_position = simulation_manager.current_position.copy()

            simulation_manager.hold()
            simulated_boards.append(np.copy(simulation_manager.board))

            # 元の状態に戻す
            simulation_manager.current_shape = temp_current_shape
            simulation_manager.hold_shape = temp_hold_shape
            simulation_manager.current_position = temp_current_position

            simulation_manager.board = np.copy(self.board)
            simulation_manager.current_shape.shape = np.copy(self.current_shape.shape)

        return np.array(simulated_boards), actions, scores


    
    
    def get_features(self, board):
        hole_count = 0
        blocks_above_holes = self.get_above_block_squared_sum()  # 新しい特徴量
        row_transitions = 0
        column_transitions = 0
        bumpiness = 0
        cumulative_wells = 0
        center_max_height = 0
        aggregate_height = 0
        heights = [0] * BOARD_WIDTH

        for x in range(BOARD_WIDTH):
            block_found = False
            for y in range(BOARD_HEIGHT):
                if board[y, x] != 0:
                    block_found = True
                    heights[x] = max(heights[x], BOARD_HEIGHT - y)
                    if BOARD_WIDTH // 2 - 2 <= x < BOARD_WIDTH // 2 + 2:
                        center_max_height = max(center_max_height, heights[x])
                        
                elif block_found:
                    hole_count += 1
            aggregate_height += heights[x]
            
        

        
        for x in range(1, BOARD_WIDTH):
            bumpiness += abs(heights[x] - heights[x - 1])

        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                if x < BOARD_WIDTH - 1:
                    if board[y, x] == 0 and board[y, x + 1] != 0:
                        row_transitions += 1
                    elif board[y, x] != 0 and board[y, x + 1] == 0:
                        row_transitions += 1

                if y < BOARD_HEIGHT - 1:
                    if board[y, x] == 0 and board[y + 1, x] != 0:
                        column_transitions += 1
                    elif board[y, x] != 0 and board[y + 1, x] == 0:
                        column_transitions += 1
                    
                
                        
        cumulative_wells = 0
        for x in range(1, BOARD_WIDTH - 1):
            well_depth = 0
            for y in range(BOARD_HEIGHT):
                if board[y, x] == 0 and board[y, x - 1] != 0 and board[y, x + 1] != 0:
                    well_depth += 1
                else:
                    if well_depth > 0:
                        cumulative_wells += well_depth * (well_depth + 1) // 2
                        well_depth = 0


        current_shape_type = self.current_shape.type
        next_shape_types = [shape.type for shape in self.next_shapes]
        hold_shape_type = self.hold_shape.type if self.hold_shape is not None else -1

        features = [
            hole_count,
            blocks_above_holes,
            self.latest_clear_mino_height,
            row_transitions,
            column_transitions,
            bumpiness,
            cumulative_wells,
            center_max_height,
            aggregate_height,
            current_shape_type
        ] + next_shape_types + [hold_shape_type]

        return features
    
    
    def get_above_block_squared_sum(self) -> int:
        # ========== above_block_squared_sum ========== #
        # 空マスで自身より上部にあるブロックの数の二乗和
        res = 0
        for i in range(BOARD_HEIGHT):
            for j in range(BOARD_WIDTH):
                if self.board[i, j] != 0:
                    continue
                cnt = 0
                for k in range(i - 1, -1, -1):
                    if self.board[k, j] != 0:
                        cnt += 1
                res += cnt**2
                
        return res
