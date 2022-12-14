import numpy as np

class MonteCarlo:
    """
    Monte Carlo Agent
    -------------------------
    Implements first visit and every visit MC agents
    
    Parameters
    -------------------------
    num_states:     number of observable states
    num_actions:    number of total possible actions 
                    (assumption => every action is possible in every state)
    visit:          "first" or "every", default value is "every"
    discount_factor: discount factor for return calculation


    alpha_mode:     "constant" or "inverse" or None
                    default None when 1/N(s) is used for Q-value updates
                    when "constant", takes the value from alpha
                    when "inverse", takes the 1/((episode no.)^alpha) for scaling
    alpha:          can be a float for constant/inverse epsilon

    """
    def __init__(self, num_states, num_actions, visit= "every", discount_factor= 0.9, 
                alpha= None, alpha_mode= None):
        self.num_states = num_states
        self.num_actions = num_actions
        self.episode_count = 0

        self.Q_values = np.random.rand(self.num_states, self.num_actions)
        # number of enocounters of state-action pair
        self.state_action_visit = np.zeros((self.num_states, self.num_actions))

        
        self.visit_mode = visit
        self.discount_factor = discount_factor


        self.alpha_mode = alpha_mode
        self.alpha = alpha
        self.alpha_power = None
        # if inverse alpha, initialize as 1, store the power of inverse seperately
        if self.alpha_mode == "inverse":
            # store the power of alpha updates
            self.alpha_power = self.alpha if self.alpha is not None else 1.0
            self.alpha = 1      # make multiplier factor 1



    def __str__(self) -> str:
        s = "Monte Carlo Agent\n"
        for key in self.__dict__:
            if key in ['num_states', 'num_actions', 'visit_mode', 'discount_factor',
                        'epsilon_mode', 'epsilon', 'alpha_mode', 'alpha']:
                        s += f"{key}: {self.__dict__[key]}\n"
        return s


    def calculateReturn (self, rewards, T):
        """Returns the return values for 0 to T-1"""
        G = np.zeros(T)
        G[T-1] = rewards[T-1]   # last return at timestep T-1
        
        idx = T-2
        while idx >= 0:
            G[idx] = rewards[idx] + self.discount_factor * G[idx+1]   # dynamically fill return
            idx -= 1
        
        return G

    def updateQ (self, episodes):
        """Update Q-values after single/multiple episode(s)"""

        is_visited_sa = [[False]*self.num_actions]*self.num_states  # for first visit MC only

        for episode in episodes:    
            self.episode_count += 1     # increment episode count

            # every episode is a list of 
            # 3 lists: states, actions, rewards
            states = episode[0]
            actions = episode[1]
            rewards = episode[2]

            T = len(states)                         # timesteps till end
            G = self.calculateReturn(rewards, T)    # pre-calculate all returns from every timestep

            # iterate through the episode for each timestep
            for t in range(T):
                S_t = states[t]         # S_t
                A_t = actions[t]        # A_t

                # increment state-action visit count according to mode of visit
                # first-visit MC
                if self.visit_mode == "first" and (not is_visited_sa[S_t][A_t]):  
                    is_visited_sa[S_t][A_t] = True  # set visited
                    self.state_action_visit[S_t][A_t] += 1

                    # forgetting enabled or not
                    if self.alpha_mode is None:
                        self.Q_values[S_t][A_t] += (G[t] - self.Q_values[S_t][A_t])/self.state_action_visit[S_t][A_t]
                    else:
                        self.Q_values[S_t][A_t] += self.alpha * (G[t] - self.Q_values[S_t][A_t])

                # every-visit MC
                elif self.visit_mode == "every":
                    self.state_action_visit[S_t][A_t] += 1

                    # forgetting enabled or not
                    if self.alpha_mode is None:
                        self.Q_values[S_t][A_t] += (G[t] - self.Q_values[S_t][A_t])/self.state_action_visit[S_t][A_t]
                    else:
                        self.Q_values[S_t][A_t] += self.alpha * (G[t] - self.Q_values[S_t][A_t])
            
            # reinit is_visited before next episode
            if self.visit_mode == "first":
                for s in range(self.num_states):
                    for a in range(self.num_actions):
                            is_visited_sa[s][a] = False
            
            # update alpha if inverse alpha is in action
            if self.alpha_mode == "inverse":
                self.alpha = 1/np.power(self.episode_count, self.alpha_power)

        



    def pickAction (self, state, epsilon):
        """ Return an action given the state"""
        """
        epsilon greedy choice of action from state
        """
        p = np.random.random()
        if p < epsilon:
            return int(np.random.choice(self.num_actions))
        else:
            return int(np.argmax(self.Q_values[state]))
        

        



        

