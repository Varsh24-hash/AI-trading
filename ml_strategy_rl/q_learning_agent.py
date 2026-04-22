import numpy as np
import random

class QLearningAgent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.q_table = {}

        self.alpha = 0.1
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    # ✅ DISCRETIZATION (CRITICAL FIX)
    def discretize(self, state):
        return tuple(np.round(state, 2))  # keep precision

    def choose_action(self, state):

        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        key = self.discretize(state)

        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.action_size)

        return np.argmax(self.q_table[key])

    def update(self, state, action, reward, next_state):

        key = self.discretize(state)
        next_key = self.discretize(next_state)

        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.action_size)

        if next_key not in self.q_table:
            self.q_table[next_key] = np.zeros(self.action_size)

        best_next = np.max(self.q_table[next_key])

        self.q_table[key][action] += self.alpha * (
            reward + self.gamma * best_next - self.q_table[key][action]
        )

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay