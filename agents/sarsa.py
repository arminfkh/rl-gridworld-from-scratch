import numpy as np


class SarsaAgent:
    def __init__(self, rows, cols, num_actions, epsilon=0.1, alpha=0.1, gamma=0.99):
        self.q_table = np.zeros((rows, cols, num_actions))
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.num_actions = num_actions

    def choose_action(self, state):
        row, col = state
        random_num = np.random.rand()

        if random_num < self.epsilon:
            chosen_action = np.random.randint(self.num_actions)

        else:
            q_values = self.q_table[row, col]
            chosen_action = np.argmax(q_values)

        return chosen_action

    def update(self, state, action, reward, next_state, next_action, terminated):
        row, col = state
        next_row, next_col = next_state
        cur_q = self.q_table[row, col, action]

        if terminated:
            next_q = 0

        else:
            next_q = self.q_table[next_row, next_col, next_action]

        target = reward + self.gamma * next_q

        new_q = cur_q + self.alpha * (target - cur_q)

        self.q_table[row, col, action] = new_q
