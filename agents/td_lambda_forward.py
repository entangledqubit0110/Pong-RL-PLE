import numpy as np

class Td_Lambda:
    """
    Td Lambda Agent
    -------------------------
    Implements Temporal Difference Learning Agent which looks 'forward' to calculate the TD errors and update the Q values
    
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

    alpha_mode:     "constant" or "inverse" or None
                    default None when 1/N(s) is used for Q-value updates
                    when "constant", takes the value from alpha
                    when "inverse", takes the 1/((episode no.)^alpha) for scaling
    alpha:          can be a float for constant/inverse epsilon

    """
    def __init__(self, num_states, num_actions, visit= "every", discount_factor= 0.9, 
                epsilon= None, epsilon_mode= None, alpha= None, alpha_mode= None,lamda=0.5):
        self.num_states = num_states
        self.num_actions = num_actions
        self.episode_count = 0

        self.Q_value = np.random.rand(self.num_states, self.num_actions)
        self.policy = np.random.choice(range(self.num_actions), size= (self.num_states))

        
        self.lamda=lamda
        self.discount_factor = discount_factor

        self.epsilon_mode = epsilon_mode
        self.epsilon = epsilon

        self.alpha_mode = alpha_mode
        self.alpha = alpha
        self.alpha_power = None
        # if inverse alpha, initialize as 1, store the power of inverse seperately
        if self.alpha_mode == "inverse":
            # store the power of alpha updates
            self.alpha_power = self.alpha if self.alpha is not None else 1.0
            self.alpha = 1      # make multiplier factor 1



    def __str__(self) -> str:
        s = "TD Agent\n"
        for key in self.__dict__:
            if key in ['num_states', 'num_actions', 'discount_factor',
                        'epsilon_mode', 'epsilon', 'alpha_mode', 'alpha', 'lamda']:
                        s += f"{key}: {self.__dict__[key]}\n"
        return s


    def calculateTDError(self, states, actions, rewards):
        """Returns the return values for 0 to T-1"""
        T = len(rewards)
        TDError = np.zeros(T)
        TDError[T-1] = rewards[T-1] - self.Q_value[states[T-1]][actions[T-1]]  # last return at timestep T-1
        
        idx = T-2
        while idx >= 0:
            delta_t = rewards[idx] + self.discount_factor * self.Q_value[states[idx]][actions[idx]] - self.Q_value[states[idx+1]][actions[idx+1]] #delta_t in slides
            TDError[idx] = delta_t + self.discount_factor * self.lamda * TDError[idx+1]   # dynamically find TD error of current state action pair
            idx -= 1
        
        return TDError

    def updateQ (self, episode):
        """Update Q-values after an episode"""  

        # every episode is a list of 
        # 3 lists: states, actions, rewards
        states = episode[0]
        actions = episode[1]
        rewards = episode[2]

        T = len(states)                         # timesteps till end
        TDError = self.calculateTDError(states, actions, rewards)    # pre-calculate the TDErrors from every timestep

        # iterate through the episode for each timestep
        for t in range(T):
            S_t = states[t]         # S_t
            A_t = actions[t]        # A_t
            #Updating the Q values using the precalculated TD errors using the rewards in future timesteps of the episode(essentially difference with the Lambda Return)
            self.Q_value[S_t][A_t] += self.alpha * TDError[t]
        
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
            if self.epsilon_mode is None:    # pure greedy choice
                self.policy[s] = np.argmax(self.Q_value[s])
            else:
                # epsilon greedy
                p = np.random.random()
                if p < self.epsilon:
                    self.policy[s] = np.random.choice(self.num_actions)
                else:
                    self.policy[s] = np.argmax(self.Q_value[s])
        



    def pickAction (self, obs):
        """ Return an action given the state"""
        return self.policy[obs]
        
        

        



        

