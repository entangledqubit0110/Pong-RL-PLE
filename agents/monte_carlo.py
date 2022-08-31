import random
import numpy as np

class MonteCarlo:
    """
    Monte Carlo Agent
    -------------------------
    Implements first visit and every visit MC agents
    """
    def __init__(self, num_states, num_actions, visit= "first", discount_factor= 0.9):
        self.num_states = num_states
        self.num_actions = num_actions

        self.V_value = []
        self.Q_value = []
        self.policy = []
        for i in range(self.num_states):                            
            # initialize Q-values as random
            # Q-value for all actions for a certain state
            Q_s = [np.random.random_sample(size= num_actions)]      
            self.Q_value.append(Q_s)
            
            # initialize V-values
            self.V_value.append(random.random())   

            # initialize policy
            self.policy.append(random.choice(range(self.num_actions)))
        
        self.visit_mode = visit
        self.discount_factor = discount_factor
    

    def calculateReturn (self, rewards, T):
        """Returns the return values for 0 to T-1"""
        G = [0]*T
        G[T-1] = rewards[T-1]   # last return at timestep T-1
        
        idx = T-2
        while idx >= 0:
            G[idx] = self.discount_factor*G[idx+1] + rewards[idx]   # dynamically fill return
            idx -= 1
        
        return G

    def update (self, episodes):
        """Update V and Q-values after single/multiple episode(s)"""
        state_visit = [0]*self.num_states       # number of encounter of a state across episodes
        is_visited = [False]*self.num_states    # for first_visit MC only

        state_action_visit = [[0]*self.num_actions]*self.num_states   # number of enocounters of state-action pair
        is_visited_sa = [[False]*self.num_actions]*self.num_states  # for first visit MC only

        total_returns_s = [0]*self.num_states       # calculate return for states in a specific episode
        total_returns_s_a = [[0]*self.num_actions]*self.num_states  # calculate return for state-action pair in specific episode

        for episode in episodes:    
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

                # increment total return for each encounter
                total_returns_s[S_t] += G[t]
                total_returns_s_a[S_t][A_t] += G[t] 

                # increment state visit count according to mode of visit
                # first-visit MC
                if self.visit_mode == "first" and (not is_visited[S_t]):  
                    is_visited[S_t] = True      # set visited
                    state_visit[S_t] += 1       # increment counter
                # every-visit MC
                elif self.visit_mode == "every":    
                    state_visit[S_t] += 1       # increment counter
                
                # increment state-action visit count also
                # first-visit MC
                if self.visit_mode == "first" and (not is_visited_sa[S_t][A_t]):  
                    is_visited_sa[S_t][A_t] = True  # set visited
                    state_action_visit[S_t][A_t] += 1
                # every-visit MC
                elif self.visit_mode == "every":
                    state_action_visit[S_t][A_t] += 1
            
            
            # incremental update after episode
            # for V-values
            for s in range(self.num_states):
                if state_visit[s] > 0:  # update for only at least once visited
                    self.V_value[s] += ((total_returns_s[s] - self.V_value[s])/state_visit[s])

                # for Q-values
                for a in range(self.num_actions):
                    if state_action_visit[s][a] > 0:    # update for only at least once visited
                        self.Q_value[s][a] += ((total_returns_s_a[s] - self.Q_value[s][a])/state_action_visit[s][a])

            # reinit is_visited and total_return before next episode
            for s in range(self.num_states):
                total_returns_s[s] = 0

                if self.visit_mode == "first":
                    is_visited[s] = False
                
                for a in range(self.num_actions):
                    total_returns_s_a[s][a] = 0

                    if self.visit_mode == "first":
                        is_visited_sa[s][a] = False

        # update policy from updated Q values
        # greedy choice
        for s in range(self.num_states):
            self.policy[s] = np.argmax(self.Q_value[s])
            
    def pickAction (self, state):
        """ Return an action given the state"""
        return self.policy[state]
        



        

