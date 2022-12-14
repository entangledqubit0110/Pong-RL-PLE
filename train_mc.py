from ple.games.pong import Pong
from ple import PLE
from discrete import Discretizer
from agents.monte_carlo import MonteCarlo
from util import getActionFromIdx
import signal

from util import print_msg_box
from pprint import pprint

# game panel details
WIDTH = 220
HEIGHT = 148
FPS = 60

# discretization params
NUM_BALL_X_BINS = 12            # important
NUM_BALL_Y_BINS = 12            # important
NUM_PLAYER_Y_BINS = 12           # important

NUM_BALL_X_VEL_BINS = 5         # less important
NUM_BALL_Y_VEL_BINS = 5         # less important

NUM_CPU_Y_BINS = 1              # ignore for now == single bin
NUM_PLAYER_VEL_BINS = 1         # ignore for now == single bin

bins = {}
bins["player_y"] = NUM_PLAYER_Y_BINS
bins["cpu_y"] = NUM_CPU_Y_BINS
bins["ball_x"] = NUM_BALL_X_BINS
bins["ball_y"] = NUM_BALL_Y_BINS
bins["ball_velocity_x"] = NUM_BALL_X_VEL_BINS
bins["ball_velocity_y"] = NUM_BALL_Y_VEL_BINS
bins["player_velocity"] = NUM_PLAYER_VEL_BINS
print_msg_box(" BINS ")
pprint(bins)

NUM_STATES =    (NUM_BALL_X_BINS*
                NUM_BALL_Y_BINS*
                NUM_PLAYER_Y_BINS*
                NUM_CPU_Y_BINS*
                NUM_BALL_X_VEL_BINS*
                NUM_BALL_Y_VEL_BINS*
                NUM_PLAYER_VEL_BINS)

NUM_ACTIONS = 3


# utility functions
def getGameStateIdx (discrete_gameState):
    """Return idx of discretized gameState, idx belongs in range 0 to NUM_STATE"""
    # get bins for every state variable
    player_y = discrete_gameState["player_y"]
    player_velocity_y = discrete_gameState["player_velocity"]
    cpu_y = discrete_gameState["cpu_y"]
    ball_x = discrete_gameState["ball_x"]
    ball_y = discrete_gameState["ball_y"]
    ball_velocity_x = discrete_gameState["ball_velocity_x"]
    ball_velocity_y = discrete_gameState["ball_velocity_y"]

    idx =   (
                player_y + NUM_PLAYER_Y_BINS*(
                    player_velocity_y + NUM_PLAYER_VEL_BINS*(
                        cpu_y + NUM_CPU_Y_BINS*(
                            ball_x + NUM_BALL_X_BINS*(
                                ball_y + NUM_BALL_Y_BINS*(
                                    ball_velocity_x + NUM_BALL_X_VEL_BINS*(
                                        ball_velocity_y
                                    )
                                )
                            )
                        )
                    )
                )
            )
    
    return idx


# initialize game
game = Pong(width= WIDTH, height= HEIGHT)
p = PLE(game, fps=FPS, display_screen=False, force_fps=True)


limits = {}
# limits for position
MIN_Y_POS = 0
MAX_Y_POS = game.height
MIN_X_POS = 0
MAX_X_POS = game.width

limits["player_y"] = (MIN_Y_POS, MAX_Y_POS)
limits["ball_x"] = (MIN_X_POS, MAX_X_POS)
limits["ball_y"] = (MIN_Y_POS, MAX_Y_POS)
limits["cpu_y"] = (MIN_Y_POS, MAX_Y_POS)

# limits for velocity
MAX_PLAYER_VELOCITY = game.agentPlayer.speed
MIN_PLAYER_VELOCITY = -1*MAX_PLAYER_VELOCITY
limits["player_velocity"] = (MIN_PLAYER_VELOCITY, MAX_PLAYER_VELOCITY)

MAX_BALL_X_VELOCITY = game.width
MIN_BALL_X_VELOCITY = -1*MAX_BALL_X_VELOCITY
limits["ball_velocity_x"] = (MIN_BALL_X_VELOCITY, MAX_BALL_X_VELOCITY)

MAX_BALL_Y_VELOCITY = game.height
MIN_BALL_Y_VELOCITY = -1*MAX_BALL_Y_VELOCITY
limits["ball_velocity_y"] = (MIN_BALL_Y_VELOCITY, MAX_BALL_Y_VELOCITY)

print_msg_box(" LIMITS ")
pprint(limits)


# agent
agent = MonteCarlo(num_states= NUM_STATES, num_actions= NUM_ACTIONS,
                    discount_factor=0.95, alpha_mode=None)
# epsilon at first 1
# decay epsilon as 1\episode_cnt
epsilon = 1

print_msg_box(" AGENT ")
print(agent)

# discretizer
dz = Discretizer(limits= limits, bins= bins)
dz.createBins()
pprint(dz.binBoundary)


episode_idx = 0
# start game
p.init()


states = []
actions = []
rewards = []

# logfile
f_reward = open("rewards.log", "w")

def handler (signum, frame):
    print("Saving current Q-values...")
    with open("final_Q.log", 'w') as fp:
        for s in range(agent.num_states):
            for a in range(agent.num_actions):
                fp.write(str(agent.Q_values[s][a]))
                fp.write(" ")
            fp.write("\n")
    exit(1)

signal.signal(signal.SIGINT, handler)

FOLLOW_REWARD_SCALING = 200

tot_reward = 0
while True: 
    if p.game_over():
        # store episode information
        episode_idx += 1    
        # epsilon for next episode
        epsilon = 1/(episode_idx+1)

        episode = [states, actions, rewards]
        tot_reward += sum(rewards)
        if episode_idx % 1000 == 0:
            print(f"episode {episode_idx-1000+1}:{episode_idx}: avg reward = {tot_reward/1000}")
            tot_reward = 0
        
        # logging
        f_reward.write(f"{episode_idx},{len(rewards)},{sum(rewards)}\n")


        # update agent
        agent.updateQ([episode])

        # reset before next episode
        states.clear()
        actions.clear()
        rewards.clear()
        p.reset_game()

    
    # state
    gameState = game.getGameState()
    discreteState = dz.discretize(gameState)
    stateIdx = getGameStateIdx(discreteState)
    states.append(stateIdx)

    # action
    actionIdx = agent.pickAction(stateIdx, epsilon= epsilon)
    action = getActionFromIdx(actionIdx)
    actions.append(actionIdx)

    # reward
    reward = p.act(action)
    # following the ball reward
    # scaled reward of negative of abs diff between ball and players y position
    follow_reward = -1*abs(gameState['ball_y'] - gameState['player_y'])/FOLLOW_REWARD_SCALING
    reward += follow_reward

    rewards.append(reward)