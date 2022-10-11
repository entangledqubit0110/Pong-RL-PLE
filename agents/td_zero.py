import numpy as np

class TD_Zero:
    """
    TD Zero agent
    """
    def __init__(self, num_states, num_actions, alpha, alpha_mode = None, epsilon = None, epsilon_mode = None, discount_factor = 0.9):
        self.num_states = num_states
        self.num_actions = num_actions
        self.episode_count = 0

        # initialize all Q values to 0
        self.Q_value = np.zeros((self.num_states, self.num_actions))

        # discount factor
        self.discount = discount_factor
        self.policy = np.random.choice(range(self.num_actions), size= (self.num_states))

        self.epsilon_mode = epsilon_mode
        self.epsilon = epsilon

        # constant alpha and epsilon
        self.alpha = alpha

        # initialize last seen state and action pair
        self.lastState = -1
        self.lastAction = -1

        self.alpha_mode = alpha_mode
        self.alpha = alpha
        self.alpha_power = None
        # if inverse alpha, initialize as 1, store the power of inverse seperately
        if self.alpha_mode == "inverse":
            # store the power of alpha updates
            self.alpha_power = self.alpha if self.alpha is not None else 1.0
            self.alpha = 1      # make multiplier factor 1


    def updateQ (self, state, action, reward):
        """
        Update Q value for lastState and lastAction from current state and action
        """
        self.Q_value[self.lastState][self.lastAction] += self.alpha * (reward
                                                        + self.discount * self.Q_value[state][action]
                                                        - self.Q_value[self.lastState][self.lastAction])
        self.episode_count += 1     # increment episode count
        
        # update alpha if inverse alpha is in action
        if self.alpha_mode == "inverse":
            self.alpha = 1/np.power(self.episode_count, self.alpha_power)
            
    def updatePolicy (self):
        """Update the policy based on current Q-values"""
        # update epsilon for next policy update
        if self.epsilon_mode == "inverse":
            self.epsilon = 1/self.episode_count

        for s in range(self.num_states):
            # epsilon greedy
            p = np.random.random()
            if p < self.epsilon:
                self.policy[s] = np.random.choice(self.num_actions)
            else:
                self.policy[s] = np.argmax(self.Q_value[s])



    def pickAction (self, obs):
        """ Return an action given the state"""
        return self.policy[obs]



