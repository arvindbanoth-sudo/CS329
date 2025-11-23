# agents.py
import numpy as np


class BaseAgent:

    def __init__(self,
                 n_states,
                 n_actions,
                 alpha = 0.1,
                 gamma = 0.99,
                 epsilon = 1.0,
                 epsilon_min = 0.05,
                 epsilon_decay = 0.995):

        self.n_states = n_states
        self.n_actions = n_actions

        self.alpha   = alpha
        self.gamma   = gamma
        self.epsilon = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-table
        self.Q = np.zeros((n_states, n_actions))


    def select_action(self, s):

        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)

        return np.argmax(self.Q[s])


    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min,
                           self.epsilon * self.epsilon_decay)



class QLearningAgent(BaseAgent):
    """ Off-policy """

    def update(self, s, a, r, s2, done):

        best_a2 = np.argmax(self.Q[s2])

        target = r
        if not done:
            target += self.gamma * self.Q[s2, best_a2]

        self.Q[s,a] += self.alpha * (target - self.Q[s,a])



class SarsaAgent(BaseAgent):
    """ On-policy """

    def update(self, s, a, r, s2, a2, done):

        target = r
        if not done:
            target += self.gamma * self.Q[s2, a2]

        self.Q[s,a] += self.alpha * (target - self.Q[s,a])
