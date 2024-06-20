import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from GameManager import GameManager

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
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.env = GameManager()
        self.update_target_model()
        
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
        target_values = rewards + self.gamma * next_values * (1 - dones)
        

        loss = self.loss_fn(current_values, target_values)
        loss = loss.clone().detach().requires_grad_(True)
        #print(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    
    
    #盤面をシミュレートして報酬の予測値が最も高い行動の盤面に移動する
    def act(self, next_state):
        rewards = []
        #print(len(next_state))
                
        for i in range(len(next_state)):
            #print(next_state[i])
            feature = self.env.get_features(next_state[i])
            #print(feature)
            rewards.append(self.predict(feature))
            
        index_of_best_action = rewards.index(max(rewards))
        
        if random.random() <= self.epsilon:
            next_board_index =  random.randrange(len(next_state))
            #print(next_board_index)
            return next_board_index, rewards[next_board_index]
        else:
            return index_of_best_action, rewards[index_of_best_action]
        

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, predict, reward, next_state, done):
        self.memory.append((state, predict, reward, next_state, done))


    def load(self, name):
        self.model.load_state_dict(torch.load(name, map_location=self.device))

    def save(self, name):
        torch.save(self.model.state_dict(), name)
