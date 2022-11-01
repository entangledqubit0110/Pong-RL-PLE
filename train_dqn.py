import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import signal

from agents.dqn.wrappers import make_env
from agents.dqn.dqn import DQNNet
from agents.dqn.replay_memory import ReplayMemory
from agents.dqn.dqn_agent import DQNAgent

# gym env
ENV = "PongNoFrameskip-v4"

# hyperparams
hyperparameters = {
    "gamma": 0.98,
    "EPSILON_BEGIN": 1.0,
    "EPSILON_END": 0.1,
    "EPSILON_DECAY": 1e5,
    "BATCH_SIZE": 32,
    "LEARNING_RATE": 1e-3,
    "TARGET_UPDATE_FREQUENCY": 1000,
    "LEARNING_START_SIZE": 10000,
    "REPLAY_SIZE": 4 * 10**4
}

MODEL_SAVE_PATH= "dqn_net.pth"



def loss_function (batch, net, target_net):
    """Calculate MSE between actual and target state-action values"""
    states, actions, rewards, dones, next_states = batch

    # actual Q values
    states_tensor = torch.tensor(states)    # batch_size * C * H * W
    q_values_tensor = net(states_tensor)    # batch_size * A
    actions_tensor = torch.tensor(actions)  # batch_size
    q_vals = q_values_tensor.gather(1, actions_tensor.long().unsqueeze(-1)).squeeze(-1)    # batch_size * 1 index, row wise


    # expected Q values
    next_states_tensor = torch.tensor(next_states)  # batch_size * C * H * W
    next_state_q_values_tensor = target_net(next_states_tensor) # batch_size * A
    
    done_tensor = torch.BoolTensor(dones)
    next_state_q_values_tensor[done_tensor] = 0.0   # zero out the terminal states q values
    
    next_state_q_vals = next_state_q_values_tensor.max(1)[0]    # greedy choice of q value over actions
                                                                # batch_size

    rewards_tensor = torch.tensor(rewards)  # batch_size
    expected_q_vals = rewards_tensor + hyperparameters["gamma"]*next_state_q_vals

    return nn.MSELoss()(q_vals, expected_q_vals)


# create env
env = make_env(ENV)
print(f"Observation space shape: {env.observation_space.shape}")
print(f"Action space size: {env.action_space.n}")

# create dqn
dqn_net = DQNNet(env.observation_space.shape, env.action_space.n)
target_dqn_net = DQNNet(env.observation_space.shape, env.action_space.n)
print(dqn_net)

# create other essentials
replay_memory = ReplayMemory(capacity= hyperparameters["REPLAY_SIZE"])
agent = DQNAgent(env= env, replay_memory= replay_memory)
epsilon = hyperparameters["EPSILON_BEGIN"]
optimizer = optim.Adam(dqn_net.parameters(), lr= hyperparameters["LEARNING_RATE"])

total_rewards = list()
frame_idx = 0
game_idx = 0
best_mean_reward = -9999    # initialized to super negative value

# logfile
f_reward = open("rewards.log", "w")

# logging
def handler (signum, frame):
    print("Saving model")
    # Save
    torch.save(dqn_net.state_dict(), MODEL_SAVE_PATH)
    exit(1)

signal.signal(signal.SIGINT, handler)


## training loop
while game_idx<1001:
    frame_idx += 1
    epsilon = max(hyperparameters["EPSILON_END"], 
                hyperparameters["EPSILON_BEGIN"] - frame_idx/hyperparameters["EPSILON_DECAY"])

    reward = agent.step(dqn_net, epsilon)

    if reward is not None:  # end of epsiode
        game_idx += 1
        total_rewards.append(reward)
        f_reward.write(f"frame {frame_idx}, game {game_idx}, replay memory size= {len(agent.replay_memory)} and reward {reward}\n")
        if game_idx % 10 == 0: 
            print(f"frame {frame_idx}, game {game_idx}, replay memory size= {len(agent.replay_memory)} and reward {reward}")
        
        # get mean reward every 100 games
        if game_idx % 100 == 0:
            mean_reward = np.mean(total_rewards[-100:])
            print(f"\nmean reward over games {len(total_rewards)-99}:{len(total_rewards)} = {mean_reward:.3f}\n")

            # update best mean reward
            if best_mean_reward < mean_reward:
                print(f"update in best mean reward {best_mean_reward:.3f} -> {mean_reward:.3f}, saving checkpoint")
                torch.save(dqn_net.state_dict(), MODEL_SAVE_PATH)
                best_mean_reward = mean_reward
        
        # update the target net params by copying
        if game_idx % 5 == 0 :
            target_dqn_net.load_state_dict(dqn_net.state_dict())

    if len(agent.replay_memory) < hyperparameters["LEARNING_START_SIZE"]:
        continue

    # # update the target net params by copying
    # if frame_idx % hyperparameters["TARGET_UPDATE_FREQUENCY"] == 0:
    #     target_dqn_net.load_state_dict(dqn_net.state_dict())
    
    
    optimizer.zero_grad()
    loss = loss_function(batch= replay_memory.sample(hyperparameters["BATCH_SIZE"]), net= dqn_net, target_net= target_dqn_net)
    loss.backward()
    optimizer.step()



    