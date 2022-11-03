import numpy as np
import torch
import gym
import wrappers
import signal
from actor import Actor
from critic import Critic

# helper function to convert numpy arrays to tensors
def t(x): return torch.tensor(x)

# Memory
# Stores results from the networks, instead of calculating the operations again from states, etc.
class Memory():
    def __init__(self):
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def add(self, log_prob, value, reward, done):
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def clear(self):
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()  
    
    def _zip(self):
        return zip(self.log_probs,
                self.values,
                self.rewards,
                self.dones)
    
    def __iter__(self):
        for data in self._zip():
            return data
    
    def reversed(self):
        for data in list(self._zip())[::-1]:
            yield data
    
    def __len__(self):
        return len(self.rewards)


env = wrappers.make_env("PongNoFrameskip-v4") 
print(f"Observation space shape: {env.observation_space.shape}")
print(f"Action space size: {env.action_space.n}")

# config
state_dim = env.observation_space.shape[0]
n_actions = env.action_space.n
actor = Actor(env.observation_space.shape, env.action_space.n)
critic = Critic(env.observation_space.shape)
adam_actor = torch.optim.Adam(actor.parameters(), lr=1e-3)
adam_critic = torch.optim.Adam(critic.parameters(), lr=1e-3)
gamma = 0.99
memory = Memory()

# train function
def train(memory, q_val):
    values = torch.stack(memory.values)
    q_vals = np.zeros((len(memory), 1))
    
    # target values are calculated backward
    # it's super important to handle correctly done states,
    # for those cases we want our to target to be equal to the reward only
    for i, (_, _, reward, done) in enumerate(memory.reversed()):
        q_val = reward + gamma*q_val*(1.0-done)
        q_vals[len(memory)-1 - i] = q_val # store values from the end to the beginning
        
    advantage = torch.Tensor(q_vals) - values
    
    critic_loss = advantage.pow(2).mean()
    adam_critic.zero_grad()
    critic_loss.backward()
    adam_critic.step()
    
    actor_loss = (-torch.stack(memory.log_probs)*advantage.detach()).mean()
    adam_actor.zero_grad()
    actor_loss.backward()
    adam_actor.step()

# # logfile
f_reward = open("rewards.log", "w")
def handler (signum, frame):
    print("Saving models")
    # Specify a path
    PATH1 = "actor_model.pth"
    PATH2 = "critic_model.pth"
    # Save
    torch.save(actor.state_dict(), PATH1)
    torch.save(actor.state_dict(), PATH2)
    exit(1)

signal.signal(signal.SIGINT, handler)

max_episode_num = 5000

episode_rewards = []
average_rewards = []
num_steps = []

###Training Loop
for episode in range(max_episode_num):
    done = False
    total_reward = 0
    state = env.reset()
    steps = 0

    while not done:
        probs = actor(t(np.array([state])))
        # dist = torch.distributions.Categorical(probs=probs)
        # action = dist.sample()

        action = np.random.choice(n_actions,p=np.squeeze(probs.detach().numpy()))
        log_prob = torch.log(probs.squeeze(0)[action])
        
        next_state, reward, done, info = env.step(action)
        
        total_reward += reward
        steps += 1
        memory.add(log_prob, critic(t(np.array([state]))), reward, done)
        
        state = next_state
        
        # train if done or num steps > max_steps
        if done:
            last_q_val = critic(t(np.array([next_state]))).detach().data.numpy()
            train(memory, last_q_val)
            memory.clear()
            num_steps.append(steps)
            f_reward.write(f"{episode},{steps},{total_reward}\n")
            if episode>0 and episode % 10 == 0:
                average_rewards.append(np.mean(episode_rewards[-10:0]))
                print(f"episode {episode-10+1}:{episode}: avg reward = {np.mean(episode_rewards[-10:0])}")
            
    episode_rewards.append(total_reward)