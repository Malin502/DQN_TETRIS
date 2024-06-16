import pygame
import sys
import time
from GameManager import GameManager, COLORS, BOARD_HEIGHT, BOARD_WIDTH
from DQNAgent import DQNAgent, action_to_string


# 定数
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
BLOCK_SIZE = 30
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

class PlayerController:
    
    def train_dqn(self, episodes):
        pygame.init()
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Tetris Training")
        clock = pygame.time.Clock()

        input_shape = (1, BOARD_HEIGHT, BOARD_WIDTH)
        num_actions = 5
        agent = DQNAgent(input_shape, num_actions)

        for e in range(episodes):
            env = GameManager()
            state = env.get_state()

            while not env.is_game_over():
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                
                action = agent.act(state)
                env.move(action_to_string(action))
                next_state = env.get_state()
                reward = env.get_score()
                done = env.is_game_over()
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                agent.replay()

                # 描画
                env.draw_board(screen, env.get_board(), env.get_current_shape(), env.get_current_shape_type(), env.get_current_position(),
                        env.get_next_shapes(), env.get_hold_shape(), env.get_hold_shape_type(), env.get_score())
                pygame.display.flip()

            print(f"Episode {e+1}/{episodes}, Score: {env.get_score()}")

        agent.save("dqn_tetris.pth")
        pygame.quit()
        
    def play_game(self):
        pygame.init()
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Tetris")
        clock = pygame.time.Clock()
        env = GameManager()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        env.move("left")
                    elif event.key == pygame.K_RIGHT:
                        env.move("right")
                    elif event.key == pygame.K_UP:
                        env.move("rotate")
                    elif event.key == pygame.K_SPACE:
                        env.move("hard_drop")
                    elif event.key == pygame.K_DOWN:
                        env.move("hold")

            if env.is_game_over():
                env.reset()
                
            env.move("down")
            env.draw_board(screen, env.get_board(), env.get_current_shape(), env.get_current_shape_type(), env.get_current_position(),
                    env.get_next_shapes(), env.get_hold_shape(), env.get_hold_shape_type(), env.get_score())
            pygame.display.flip()
            clock.tick(5)

        pygame.quit()