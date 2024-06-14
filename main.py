import pygame
import sys
from GameManager import GameManager, COLORS

# 定数
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
BLOCK_SIZE = 30
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

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
    
    

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Tetris")
    clock = pygame.time.Clock()
    game_manager = GameManager()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    game_manager.move("left")
                elif event.key == pygame.K_RIGHT:
                    game_manager.move("right")
                elif event.key == pygame.K_DOWN:
                    game_manager.move("down")
                elif event.key == pygame.K_UP:
                    game_manager.rotate()
                elif event.key == pygame.K_SPACE:
                    game_manager.hold()
                elif event.key == pygame.K_h:
                    game_manager.hard_drop()

        game_manager.move("down")
        draw_board(screen, game_manager.get_board(), game_manager.get_current_shape(), game_manager.get_current_shape_type(), game_manager.get_current_position(), 
                   game_manager.get_next_shapes(), game_manager.get_hold_shape(), game_manager.get_hold_shape_type(), game_manager.get_score())
        pygame.display.flip()
        clock.tick(4)

if __name__ == "__main__":
    main()
