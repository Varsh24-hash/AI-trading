import numpy as np
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim


# =========================
# 🧠 Neural Network
# =========================

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        return self.model(x)


# =========================
# 🤖 RL Agent
# =========================

class RLAgent:

    def __init__(self, state_size, action_size):

        self.state_size = state_size
        self.action_size = action_size

        self.memory = deque(maxlen=2000)

        # Hyperparameters
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    # =========================
    # 🧠 Store experience
    # =========================
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # =========================
    # 🎯 Choose action
    # =========================
    def act(self, state):

        if np.random.rand() <= self.epsilon:
            return random.choice([0, 1, 2])  # HOLD, BUY, SELL

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.model(state)

        return torch.argmax(q_values).item()

    # =========================
    # 🔁 Train from memory
    # =========================
    def replay(self, batch_size=32):

        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in batch:

            state = torch.FloatTensor(state).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)

            target = reward

            if not done:
                target = reward + self.gamma * torch.max(self.model(next_state)).item()

            target_f = self.model(state)
            target_f = target_f.clone().detach()

            target_f[action] = target

            output = self.model(state)

            loss = self.criterion(output, target_f)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# =========================
# 🚀 Training Loop
# =========================

def train_agent(env, episodes=50):

    state_size = len(env.reset())
    action_size = 3  # HOLD, BUY, SELL

    agent = RLAgent(state_size, action_size)

    for e in range(episodes):

        state = env.reset()
        total_reward = 0

        while True:
            action_idx = agent.act(state)

            # map action index → actual action
            action_map = {0: 0, 1: 1, 2: -1}
            action = action_map[action_idx]

            next_state, reward, done, _ = env.step(action)

            agent.remember(state, action_idx, reward, next_state, done)

            state = next_state
            total_reward += reward

            if done:
                break

            agent.replay()

        print(f"Episode {e+1}/{episodes}, Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

    return agent