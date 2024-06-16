import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from GameManager import GameManager, BOARD_HEIGHT, BOARD_WIDTH

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(32 * (input_shape[1] - 2) * (input_shape[2] - 2), 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_actions)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, input_shape, num_actions):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(input_shape, num_actions).to(self.device)
        self.target_model = DQN(input_shape, num_actions).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters())
        self.loss_fn = nn.MSELoss()
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.update_target_steps = 1000
        self.steps = 0

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 4)
        state = state.to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values, dim=1).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.cat(states).to(self.device)
        actions = torch.tensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        next_states = torch.cat(next_states).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        current_q_values = self.model(states).gather(1, actions).squeeze()
        next_q_values = self.target_model(next_states).max(1)[0]
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = self.loss_fn(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.steps += 1
        if self.steps % self.update_target_steps == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.model.state_dict(), path)

def action_to_string(action):
    if action == 0:
        return "left"
    elif action == 1:
        return "right"
    elif action == 2:
        return "down"
    elif action == 3:
        return "rotate"
    elif action == 4:
        return "hard_drop"
    return "none"

def train_dqn(episodes):
    env = GameManager()
    input_shape = (1, BOARD_HEIGHT, BOARD_WIDTH)
    num_actions = 5
    agent = DQNAgent(input_shape, num_actions)

    for e in range(episodes):
        env = GameManager()
        state = env.get_state()
        total_reward = 0

        while not env.is_game_over():
            action = agent.act(state)
            env.move(action_to_string(action))
            next_state = env.get_state()
            reward = env.get_reward()
            total_reward += reward
            done = env.is_game_over()
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.replay()

        print(f"Episode {e+1}/{episodes}, Total Reward: {total_reward}")

    agent.save("dqn_tetris.pth")