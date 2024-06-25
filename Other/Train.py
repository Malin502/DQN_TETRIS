import argparse
import os
import shutil
from random import random, randint, sample

import numpy as np
import torch
import torch.nn as nn
from Agent import DeepQNetwork
from Tetris import Tetris
from collections import deque

import pygame

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")
    parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    parser.add_argument("--height", type=int, default=20, help="The common height for all images")
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    parser.add_argument("--batch_size", type=int, default=512, help="The number of images per batch")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=1)
    parser.add_argument("--final_epsilon", type=float, default=1e-3)
    parser.add_argument("--num_decay_epochs", type=float, default=3000)
    parser.add_argument("--num_epochs", type=int, default=4000)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--replay_memory_size", type=int, default=15000,
                        help="Number of epoches between testing phases")
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="trained_models")

    args = parser.parse_args()
    return args

def train(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
        
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    
    
    env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size)
    model = DeepQNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.MSELoss()


    state = env.reset()
    
    
    if torch.cuda.is_available():
        model.cuda()
        state = state.cuda()

    upper_replay_memory = deque(maxlen=opt.replay_memory_size)
    lower_replay_memory = deque(maxlen=opt.replay_memory_size)
    
    epoch = 0
    frame = 0

    pygame.init()
    clock = pygame.time.Clock()

    while epoch < opt.num_epochs:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        next_steps = env.get_next_states()
        # Exploration or exploitation
        epsilon = opt.final_epsilon + (max(opt.num_decay_epochs - epoch, 0) * (
                opt.initial_epsilon - opt.final_epsilon) / opt.num_decay_epochs)
        u = random()
        
        random_action = u <= epsilon
        next_actions, next_states = zip(*next_steps.items())
        
        next_states = torch.stack(next_states)
        if torch.cuda.is_available():
            next_states = next_states.cuda()
        model.eval()
        with torch.no_grad():
            predictions = model(next_states)[:, 0]
            
        model.train()
        if random_action:
            index = randint(0, len(next_steps) - 1)
        else:
            index = torch.argmax(predictions).item()

        next_state = next_states[index, :]
        action = next_actions[index]

        reward, done = env.step(action)

        if torch.cuda.is_available():
            next_state = next_state.cuda()
        
        
        y_pos = env.current_pos["y"]
        print(y_pos) #ずっと0
        
        if y_pos < opt.height // 2:
            upper_replay_memory.append([state, reward, next_state, done])
        else:
            lower_replay_memory.append([state, reward, next_state, done])
        
        
        cleared_lines = env.get_cleared_lines()
        if cleared_lines > 100:
            done = True
        
        if done:
            final_score = env.score
            final_tetrominoes = env.tetrominoes
            final_cleared_lines = env.cleared_lines
            state = env.reset()
            if torch.cuda.is_available():
                state = state.cuda()
        else:
            state = next_state
            continue
        
        
        if len(upper_replay_memory) < opt.replay_memory_size / 10 or len(lower_replay_memory) < opt.replay_memory_size / 10:
            continue
        
        epoch += 1
        all_replay_memory = upper_replay_memory + lower_replay_memory
        batch = sample(all_replay_memory, min(len(all_replay_memory), opt.batch_size))
        state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        state_batch = torch.stack(tuple(state for state in state_batch))
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
        next_state_batch = torch.stack(tuple(state for state in next_state_batch))


        if torch.cuda.is_available():
            state_batch = state_batch.cuda()
            reward_batch = reward_batch.cuda()
            next_state_batch = next_state_batch.cuda()


        q_values = model(state_batch)
        model.eval()
        with torch.no_grad():
            next_prediction_batch = model(next_state_batch)
        model.train()

        y_batch = torch.cat(
            tuple(reward if done else reward + opt.gamma * prediction for reward, done, prediction in
                  zip(reward_batch, done_batch, next_prediction_batch)))[:, None]

        optimizer.zero_grad()
        loss = criterion(q_values, y_batch)
        loss.backward()
        optimizer.step()

        print(f"Epoch: {epoch}/{opt.num_epochs}, Score: {final_score}")

        if epoch > 0 and epoch % opt.save_interval == 0:
            torch.save(model.state_dict(), "MyModel.pth")

        # 描画のタイミングを調整
        if frame % 10 == 0:
            env.render()
            pygame.display.flip()
            clock.tick(60)  # フレームレートを設定
        frame += 1

    torch.save(model.state_dict(), "MyModel.pth")
    pygame.quit()

if __name__ == "__main__":
    opt = get_args()
    train(opt)
