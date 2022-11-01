import numpy as np
import torch
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'done', 'next_state'))


class DQNAgent:
    """
    DQN agent implementation
    """
    def __init__ (self, env, replay_memory):
        self.env = env
        self.replay_memory = replay_memory
        self._clear()
        self.last_action = 0

    def _clear (self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    def step (self, dqn_net, epsilon= 0.0):
        """
        Select an action
        Take action and one step in the environment
        Add the transition in experience replay
        """
        episode_total_reward = None

        # epsilon greedy choice
        if np.random.random() < epsilon:
            action = self.env.action_space.sample() # pick random action
        else:
            # state :: C * H * W
            state_tensor = torch.tensor(np.array([self.state]))     # 1 * C * H * W
            q_vals_tensor = dqn_net(state_tensor)   # 1 * A (A = action space size)
            _, act_tensor = torch.max(q_vals_tensor, dim=1)
            action = int(act_tensor.item())
        
        # take action
        next_state, reward, done, _ = self.env.step(action)
        self.total_reward += reward

        # store experience
        self.replay_memory.append(self.state, action, reward, done, next_state)

        # update state
        self.state = next_state

        # handle episode end        
        if done:
            episode_total_reward = self.total_reward
            self._clear()
        
        return episode_total_reward

            


