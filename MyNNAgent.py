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
    def __init__(self, state, predict, reward, next_state, done, clear_lines):
        self.state = state
        self.predict = predict
        self.reward = reward
        self.next_state = next_state
        self.done = done
        self.clear_lines = clear_lines
        
        
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
    ) -> List[Tuple[np.ndarray, float, float, np.ndarray, bool]]:
        idx = np.random.choice(len(self.buffer), size, replace=False)
        return [self.buffer[i] for i in idx]
    
    def len(self):
        return len(self.buffer)


class MyNN(nn.Module):
    def __init__(self, input_dim):
        super(MyNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 1) #報酬の予測値を返す

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
        self.batch_size = 256
        self.epochs = 8
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MyNN(state_dim).to(self.device)
        self.target_model = MyNN(state_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.num_learn = 0
        self.env = GameManager()
        self.update_target_model()
        self.last_action = None
        
        self.lower_experience_buffer = ExperienceBuffer()
        self.upper_experience_buffer = ExperienceBuffer()
        
        
    #報酬の予測値を返す
    def predict(self, feature):
        feature = torch.tensor(feature).float().to(self.device)
        return self.model(feature)
        
        
    def replay(self):
        
        
        if (self.lower_experience_buffer.len() < self.batch_size // 2
                or self.upper_experience_buffer.len() < self.batch_size - self.batch_size // 2):
            '''
            print("lower experience buffer size: ", self.lower_experience_buffer.len())
            print("upper experience buffer size: ",self.upper_experience_buffer.len(),"\n",)
            '''
            return
        
        lower_batch = self.lower_experience_buffer.sample(self.batch_size // 2)
        upper_batch = self.upper_experience_buffer.sample(self.batch_size - self.batch_size // 2)
        all_batch = lower_batch + upper_batch
        
        
        feature = np.array([sample.state for sample in all_batch])
        next_feature = np.array([sample.next_state for sample in all_batch])
        cancat_states_tensor = (torch.tensor(np.concatenate([feature,next_feature])).float().to(self.device))
        
        all_targets = self.model(cancat_states_tensor)
        
        targets = all_targets[:self.batch_size]
        next_targets = all_targets[self.batch_size:]
        

        for i, sample in enumerate(all_batch):
            targets[i] = sample.reward
            
            if not sample.done:
                targets[i] += self.gamma * next_targets[i]
                
        feature_tensor = torch.tensor(feature).float().to(self.device)
        targets_tensor = torch.tensor(targets).float().to(self.device)


        for _ in range(self.epochs):
            self.optimizer.zero_grad()
            
            predict = self.model(feature_tensor)
            loss = self.loss_fn(predict, targets_tensor)
            loss = loss.clone().detach().requires_grad_(True)
            #print(loss)
            
            loss.backward()
            self.optimizer.step()
            
            self.num_learn += 1
        
        #print("loss: ", loss.item())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        if self.num_learn >= 1700:
            self.update_target_model()
            self.num_learn = 0
            print("Data line cnt lower: ", self.lower_experience_buffer.data_line_cnt)
            print("Data line cnt upper: ", self.upper_experience_buffer.data_line_cnt)
            
            print("target model updated")
            
    
    
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
                line_clear_actions.append((i, self.predict(feature))) #index, rewardを格納
                
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
    
    
    def act_for_test(self, next_state):
        rewards = []
        
        for i in range(len(next_state)):
            feature = self.env.get_features(next_state[i])
            rewards.append(self.predict(feature))
            
        index_of_best_action = rewards.index(max(rewards))
        
        # ホールドの連続選択を回避
        if self.last_action == 'hold' and random.random() < 0.10:
            # ランダムに他の行動を選択
            next_board_index = random.choice([i for i in range(len(next_state)) if i != index_of_best_action])
            self.last_action = 'other'
        else:
            next_board_index = index_of_best_action
            self.last_action = 'hold' if index_of_best_action == len(next_state) - 1 else 'other'  
            
        return next_board_index, rewards[next_board_index]
            
            

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, predict, reward, next_state, done, y_position, line_clears):
        bufferItem = BufferItem(state, predict, reward, next_state, done, line_clears)
        
        if (y_position > 9):
            self.lower_experience_buffer.append(bufferItem)
        else:
            self.upper_experience_buffer.append(bufferItem)
        
        #print(y_position)


    def load(self, name):
        self.model.load_state_dict(torch.load(name, map_location=self.device))

    def save(self, name):
        torch.save(self.model.state_dict(), name)
