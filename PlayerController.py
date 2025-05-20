import pygame
import sys
import time

import itertools
from GameManager import GameManager, BOARD_HEIGHT, BOARD_WIDTH, BUTTON_HEIGHT, BUTTON_WIDTH
from MyNNAgent import MyNNAgent


# 定数
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
BLOCK_SIZE = 30
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

TARGET_UPDATE_FREQUENCY = 20

class PlayerController:
    
    
    def train_MyNN(self, episodes):
        pygame.init()
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Tetris Training")
        clock = pygame.time.Clock()

        env = GameManager()
        input_dim = 9
        
        agent = MyNNAgent(input_dim)
        #agent.load("MyNN3.pth")
        
        save_button_rect = pygame.Rect(SCREEN_WIDTH - BUTTON_WIDTH - 10, SCREEN_HEIGHT - BUTTON_HEIGHT - 10, BUTTON_WIDTH, BUTTON_HEIGHT)
        training = True
        
        for e in range(episodes):
            if not training:
                break
            
            env.reset()
            total_reward = 0
            total_lines_cleared = 0
            prev_score = 0
            
            
            while not env.game_over():
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                    
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        if env.is_button_clicked(event.pos, save_button_rect):
                            agent.save("mynn_tetri.pth")
                            print("Model saved")
                            #training = False
                            
                past_state = env.board.copy()
                past_feature = env.get_features(past_state)
                
                #次の盤面をシミュレートしエージェントに渡す
                next_state, actions, scores, latest_y_position = env.simulate_next_boards()
                best_index, predict_reward = agent.act(next_state, scores, latest_y_position)
                action = actions[best_index]
                
                y_position = env.action(action)
                #print(env.board)
                
                current_score = env.get_score()
                reward = current_score - prev_score  # 報酬の計算
                lines_cleared = env.lines_cleard(reward)
                total_lines_cleared += lines_cleared
                
                prev_score = current_score
                total_reward += reward
                done = env.game_over()
                
                if done:
                    reward -= 3
                    total_reward += reward
                    
                if total_lines_cleared >= 500:
                    done = True
                    
                agent.remember(past_feature, predict_reward, reward, env.get_features(env.board), done, y_position, lines_cleared)


                # 描画
                env.draw_board(screen)
                pygame.display.flip()
                
            agent.replay()

            total_reward += 3
            print(f"Episode {e+1}/{episodes}, Total Reward: {total_reward}")

        agent.save("MyNN_tetris.pth")
        pygame.quit()
        
        
        

    
    def test_MyNN(self):
        pygame.init()
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Tetris Training")
        clock = pygame.time.Clock()

        env = GameManager()
        input_dim = 12
        
        agent = MyNNAgent(input_dim)
        agent.load("NN6.pth")
        
        save_button_rect = pygame.Rect(SCREEN_WIDTH - BUTTON_WIDTH - 10, SCREEN_HEIGHT - BUTTON_HEIGHT - 10, BUTTON_WIDTH, BUTTON_HEIGHT)
        
        while True:
            
            prev_score = 0
                
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if env.is_button_clicked(event.pos, save_button_rect):
                        break
            
            past_state = env.board.copy()
            past_feature = env.get_features(past_state)
            
            #次の盤面をシミュレートしエージェントに渡す
            next_state, actions, scores, latest_y_psition= env.simulate_next_boards()
            best_index, predict_reward = agent.act_for_test(next_state)
            action = actions[best_index]
            
            y_position = env.action(action)
            
            current_score = env.get_score()
            reward = current_score - prev_score  # 報酬の計算
            prev_score = current_score
            done = env.game_over()
                
            
            if done:
                env.reset()

            # 描画
            env.draw_board(screen)
            pygame.display.flip()
                
                
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
                    elif event.key == pygame.K_h:
                        feature = env.get_hole_count(env.board)
                        print(feature)

            if env.game_over():
                env.reset()
                
            #env.move("down")
            env.draw_board(screen)
            pygame.display.flip()
            clock.tick(10)

        pygame.quit()