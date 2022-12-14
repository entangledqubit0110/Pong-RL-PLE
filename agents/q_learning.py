import numpy as np

class Q_Learning:
    """
    Q_Learning agent
    """
    def __init__(self, num_states, num_actions, alpha, discount_factor = 0.9):
        self.num_states = num_states
        self.num_actions = num_actions
        self.episode_count = 0

        # initialize all Q values to 0
        self.Q_values = np.zeros((self.num_states, self.num_actions))

        # discount factor
        self.discount = discount_factor

        # constant alpha and epsilon
        self.alpha = alpha

        # initialize last seen state and action pair
        # used for on-policy updates
        self.lastState = -1
        self.lastAction = -1

    def updateQ (self, state, action, reward):
        """
        Update Q value for lastState and lastAction from current state and action
        """
        self.Q_values[self.lastState][self.lastAction] += self.alpha * (reward
                                                        + self.discount * self.Q_values[state][np.argmax(self.Q_values[state])]
                                                        - self.Q_values[self.lastState][self.lastAction])



    def pickAction (self, state, epsilon):
        """
        epsilon greedy choice of action from state
        """
        p = np.random.random()
        if p < epsilon:
            return int(np.random.choice(self.num_actions))
        else:
            return int(np.argmax(self.Q_values[state]))




