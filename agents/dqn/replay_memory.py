import numpy as np
from collections import namedtuple, deque
import random

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'done', 'next_state'))

class ReplayMemory :
    """
    Simple Replay Buffer implemented as a dequeue
    every entry is a transition
    """
    def __init__ (self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def __len__ (self):
        return len(self.memory)

    def append (self, state, action, reward, done, next_state):
        """Add a new transition"""
        t = Transition(state, action, reward, done, next_state)
        self.memory.append(t)
    
    def sample (self, size):
        """Sample size number of elements at random"""
        tuples = random.sample(self.memory, size)
        states, actions, rewards, dones, next_states = ([] for i in range(5))
        for item in tuples:
            states.append(item.state)
            actions.append(item.action)
            rewards.append(item.reward)
            dones.append(item.done)
            next_states.append(item.next_state)
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.uint8), np.array(next_states)