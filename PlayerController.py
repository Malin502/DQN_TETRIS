import pygame
import sys
import time

import itertools
from GameManager import GameManager, BOARD_HEIGHT, BOARD_WIDTH, BUTTON_HEIGHT, BUTTON_WIDTH
from DQNAgent import DQNAgent


# 定数
SCREEN_WIDTH = 600
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

        env = GameManager()
        state_dim = BOARD_WIDTH * BOARD_HEIGHT + 5  # ボードの状態とテトリミノの種類
        action_dim = (BOARD_WIDTH * 4) + 1  # 4つの回転のそれぞれに対して横方向の移動とホールド
        agent = DQNAgent(state_dim, action_dim)
        
        save_button_rect = pygame.Rect(SCREEN_WIDTH - BUTTON_WIDTH - 10, SCREEN_HEIGHT - BUTTON_HEIGHT - 10, BUTTON_WIDTH, BUTTON_HEIGHT)
        training = True

        for e in range(episodes):
            if not training:
                break
            
            env.reset()
            state = env.get_features()
            total_reward = 0
            prev_score = 0

            while not env.game_over():
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                    
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        if env.is_button_clicked(event.pos, save_button_rect):
                            agent.save("dqn_tetris.pth")
                            training = False
                
                if not training:
                    break

                action = agent.act(state)
                if action == action_dim - 1:
                    env.hold()
                else:
                    x_move = action // 4 - BOARD_WIDTH // 2
                    rotations = action % 4
                    for _ in range(rotations):
                        env.rotate()
                    if x_move < 0:
                        for _ in range(abs(x_move)):
                            env.move("left")
                    else:
                        for _ in range(x_move):
                            env.move("right")
                    env.hard_drop()

                next_state = env.get_features().flatten()
                current_score = env.get_score()
                reward = current_score - prev_score  # 報酬の計算
                prev_score = current_score
                total_reward += reward
                done = env.game_over()
                if done:
                    reward = -3
                    
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                agent.replay()

                # 描画
                env.draw_board(screen)
                pygame.display.flip()

            print(f"Episode {e+1}/{episodes}, Total Reward: {total_reward}")

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
                        env.move("down")

            if env.game_over():
                env.reset()
                
            #env.move("down")
            env.draw_board(screen)
            pygame.display.flip()
            clock.tick(10)

        pygame.quit()