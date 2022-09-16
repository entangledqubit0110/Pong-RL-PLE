import numpy as np

class TD_Lambda:
    """
    TD Lambda Agent - Backward View 
    -------------------------
    Implements TD Agent using eligibility trace to find lambda return 
    
    Parameters
    -------------------------
    num_states:     number of observable states
    num_actions:    number of total possible actions 
                    (assumption => every action is possible in every state)
    lamda:         used for giving weights to n step returns 
    discount_factor: discount factor for return calculation

    epsilon_mode:   "constant" or "inverse" or None
                    default None when we use pure greedy policy
                    when "constant", takes the value of epsilon
                    when "inverse, uses 1/episode no. that follows GLIE property
    epsilon:        can be a float value for constant epsilon

    alpha_mode:     "constant" or "inverse"
                    when "constant", takes the value from alpha
                    when "inverse", takes the 1/((episode no.)^alpha) for scaling
    alpha:          can be a float for constant/inverse alpha

    """
    def __init__(self, num_states, num_actions, visit= "every", discount_factor= 0.9, 
                epsilon= None, epsilon_mode= None, alpha= None, alpha_mode= "inverse",lamda=0.5):
        self.num_states = num_states
        self.num_actions = num_actions
        self.episode_count = 0

        self.Q_value = np.random.rand(self.num_states, self.num_actions)
        #Accumulating td-errors for each state-action pair for an episode
        self.TD_error = np.zeros((self.num_states, self.num_actions))
        #Eligibility trace for each state action pair
        self.ETrace = np.zeros((self.num_states, self.num_actions))
        
        self.policy = np.random.choice(range(self.num_actions), size= (self.num_states))

        self.lamda=lamda
        self.discount_factor = discount_factor

        self.epsilon_mode = epsilon_mode
        self.epsilon = epsilon

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



    def __str__(self) -> str:
        s = "Backward TD Agent\n"
        for key in self.__dict__:
            if key in ['num_states', 'num_actions', 'discount_factor',
                        'epsilon_mode', 'epsilon', 'alpha_mode', 'alpha', 'lamda']:
                        s += f"{key}: {self.__dict__[key]}\n"
        return s

    def accumulateError(self, curState, curAction, reward):
        """Accumulates error for each state-action pair online at every step of an episode"""
        self.ETrace[self.lastState][self.lastAction] += 1
        for state in range(self.num_states):
            for action in range(self.num_actions):
                self.TD_error[state][action] += (reward + self.discount_factor * self.Q_value[curState][curAction]-self.Q_value[self.lastState][self.lastAction])*self.ETrace[state][action]
                self.ETrace[state][action] *= self.discount_factor * self.lamda
    
    def updateQ (self):
        """Update Q-values after an episode"""

        #Adding the TDErrors accumulated during an episode to each state-action pair Q-value
        for state in range(self.num_states):
            for action in range(self.num_actions):
                self.Q_value[state][action] += self.alpha * self.TD_error[state][action]      
        
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