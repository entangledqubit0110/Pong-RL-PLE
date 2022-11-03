import numpy as np
import torch
import gym
import wrappers
import signal
from actor import Actor
from critic import Critic


# helper function to convert numpy arrays to tensors
def t(x): return torch.tensor(x)


env = wrappers.make_env("PongNoFrameskip-v4") #,render_mode="rgb_array"
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
        advantage = reward + (1-done)*gamma*critic(t(np.array([next_state]))) - critic(t(np.array([state])))
        total_reward += reward
        steps += 1
        
        state = next_state
        
        critic_loss = advantage.pow(2).mean()
        adam_critic.zero_grad()
        critic_loss.backward()
        adam_critic.step()

        actor_loss = -log_prob*advantage.detach()
        adam_actor.zero_grad()
        actor_loss.backward()
        adam_actor.step()

    num_steps.append(steps)
    f_reward.write(f"{episode},{steps},{total_reward}\n")
    if episode!=0 and episode % 10 == 0:
        average_rewards.append(np.mean(episode_rewards[-10:0]))
        print(f"episode {episode-10+1}:{episode}: avg reward = {np.mean(episode_rewards[-10:0])}")        
    episode_rewards.append(total_reward)