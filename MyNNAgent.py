import torch
torch.manual_seed(42)
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from typing import List, Tuple
from collections import deque
from GameManager import GameManager


def lines_cleard(score):
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
    
    
class BufferItem:
    def __init__(self, state, predict, reward, next_state, done):
        self.state = state
        self.predict = predict
        self.reward = reward
        self.next_state = next_state
        self.done = done
        
        
class ExperienceBuffer:
    def __init__(self, buffer_size = 10000):
        self.buffer = deque(maxlen=buffer_size)
        self.data_line_cnt = [0, 0, 0, 0, 0]
        
    def append(self, experience: BufferItem):
        if len(self.buffer) >= self.buffer.maxlen:
            pop_item = self.buffer.popleft()
            self.data_line_cnt[pop_item.clear_lines] -= 1
            
        self.buffer.append(experience)
        self.data_line_cnt[experience.clear_lines] += 1
        
    def sample(
        self, size: int
    ) -> List[Tuple[np.ndarray, int, float, np.ndarray, bool]]:
        idx = np.random.choice(len(self.buffer), size, replace=False)
        return [self.buffer[i] for i in idx]
    
    def __len__(self):
        return len(self.buffer)


class MyNN(nn.Module):
    def __init__(self, input_dim):
        super(MyNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 1) #出力は報酬の期待値

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

class MyNNAgent:
    def __init__(self, state_dim, ):
        self.state_dim = state_dim
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MyNN(state_dim).to(self.device)
        self.target_model = MyNN(state_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.env = GameManager()
        self.update_target_model()
        self.last_action = None
        
        self.lower_experinece_buffer = ExperienceBuffer()
        self.upper_experinece_buffer = ExperienceBuffer()
        
        
    #報酬の予測値を返す
    def predict(self, feature):
        feature = torch.tensor(feature).float().to(self.device)
        return self.model(feature)
        
        
    def replay(self):
        
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        feature, predict, rewards, next_feature, dones = zip(*minibatch)
        
        #print(next_states)
        

        feature = torch.FloatTensor(feature).to(self.device)
        predict = torch.FloatTensor(predict).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        next_feature = torch.FloatTensor(next_feature).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        current_values = predict
        next_values = self.target_model(feature)
        next_values = torch.tensor(next_values).to(self.device)
        target_values = rewards + self.gamma * next_values
        

        loss = self.loss_fn(current_values, target_values)
        loss = loss.clone().detach().requires_grad_(True)
        #print(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    
    
    #盤面をシミュレートして報酬の予測値が最も高い行動の盤面に移動する
    def act(self, next_state, scores):
        #print(scores)
        rewards = []
        line_clear_actions = []
        #print(len(next_state))
        
        #ε-greedy法でランダムに行動する
        if random.random() <= self.epsilon:
            next_board_index =  random.randrange(len(next_state))
            #print(next_board_index)
            return next_board_index, self.predict(self.env.get_features(next_state[next_board_index]))
        
        
        
        for i in range(len(next_state)):
            if lines_cleard(scores[i]) >= 3:
                feature = self.env.get_features(next_state[i])
                return i, self.predict(feature)
            
            elif lines_cleard(scores[i]) >= 1:
                feature = self.env.get_features(next_state[i])
                line_clear_actions.append((i, self.predict(feature)))
                
            else:
                feature = self.env.get_features(next_state[i])
                rewards.append(self.predict(feature))
                
        if len(line_clear_actions) > 0:
                index_of_best_action = random.choice(line_clear_actions)
                return index_of_best_action[0], index_of_best_action[1]
            
        index_of_best_action = rewards.index(max(rewards))
            
        # ホールドの連続選択を回避
        if self.last_action == 'hold' and random.random() < 0.50: #学習効率化のためにホールドの連続選択を回避
            # ランダムに他の行動を選択
            next_board_index = random.choice([i for i in range(len(next_state)) if i != index_of_best_action])
            self.last_action = 'other'
        else:
            next_board_index = index_of_best_action
            self.last_action = 'hold' if index_of_best_action == len(next_state) - 1 else 'other'  
        return next_board_index, rewards[next_board_index]
            

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, predict, reward, next_state, done):
        self.memory.append((state, predict, reward, next_state, done))


    def load(self, name):
        self.model.load_state_dict(torch.load(name, map_location=self.device))

    def save(self, name):
        torch.save(self.model.state_dict(), name)
