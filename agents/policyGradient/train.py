import time
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import signal

import wrappers 
from policyNetwork import PolicyNetwork 

GAMMA = 0.95

def update_policy(policy_network, rewards, log_probs):
    discounted_rewards = []

    for t in range(len(rewards)):
        Gt = 0 
        pw = 0
        for r in rewards[t:]:
            Gt = Gt + GAMMA**pw * r
            pw = pw + 1
        discounted_rewards.append(Gt)
        
    discounted_rewards = torch.tensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9) # normalize discounted rewards
    
    policy_network.optimizer.zero_grad()
    policy_gradient = (-torch.stack(log_probs)*discounted_rewards.detach()).mean()
    policy_gradient.backward()
    policy_network.optimizer.step()



env = wrappers.make_env("PongNoFrameskip-v4") #,render_mode="rgb_array"
print(f"Observation space shape: {env.observation_space.shape}")
print(f"Action space size: {env.action_space.n}")

policy_net=PolicyNetwork(env.observation_space.shape, env.action_space.n)
print(policy_net)

# # logfile
f_reward = open("rewards.log", "w")
def handler (signum, frame):
    print("Saving model")
    # Specify a path
    PATH = "state_dict_model.pth"
    # Save
    torch.save(policy_net.state_dict(), PATH)
    exit(1)

signal.signal(signal.SIGINT, handler)


max_episode_num = 5000
# max_steps = 10000
numsteps = []
avg_numsteps = []
all_rewards = []
tot_reward = 0


for episode in range(max_episode_num):
    state = env.reset()
    log_probs = []
    rewards = []
    step = 0

    while True:
        # env.render(mode="rgb_array")          # May or may not render
        action, log_prob = policy_net.get_action(state)
        new_state, reward, done, _ = env.step(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        step = step+1
        if done:
            update_policy(policy_net, rewards, log_probs)
            numsteps.append(step)
            avg_numsteps.append(np.mean(numsteps[-10:]))
            all_rewards.append(np.sum(rewards))
            tot_reward += sum(rewards)
            f_reward.write(f"{episode},{len(rewards)},{sum(rewards)}\n")
            if episode % 50 == 0:
                print(f"episode {episode-50+1}:{episode}: avg reward = {tot_reward/50}")
                tot_reward = 0
            break
        state = new_state
        




