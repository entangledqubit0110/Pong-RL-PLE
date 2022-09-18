import numpy as np
from ple.games.pong import Pong
from ple import PLE
from discrete import Discretizer
from agents.td_zero import TD_Zero
from util import getActionFromIdx
import argparse

from util import print_msg_box
from pprint import pprint

parser = argparse.ArgumentParser()
parser.add_argument('--param_file', required= True,
                                help= "file containing saved Q values")
args = parser.parse_args()

# game panel details
WIDTH = 220
HEIGHT = 148
FPS = 60

# discretization params
NUM_BALL_X_BINS = 10           # important
NUM_BALL_Y_BINS = 10           # important
NUM_PLAYER_Y_BINS = 10           # important

NUM_BALL_X_VEL_BINS = 4         # less important
NUM_BALL_Y_VEL_BINS = 4         # less important

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
    
    return int(idx)






# initialize game
game = Pong(width= WIDTH, height= HEIGHT)
p = PLE(game, fps=FPS, display_screen=True, force_fps=False)


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

# greedy for test
epsilon = 1

# agent
agent = TD_Zero(NUM_STATES, NUM_ACTIONS, 
                alpha= 0.1, discount_factor= 0.95,epsilon=1)


# load Q values
matrix = []
with open(args.param_file, 'r') as saved_q:
    for line in saved_q:
        numbers = [np.float64(num) for num in line.split()]
        matrix.append(numbers)

matrix = np.array(matrix)
if matrix.shape != agent.Q_value.shape:
    raise ValueError("mismatching shape for saved Q values")
else:
    agent.Q_value = matrix
    

print_msg_box(" AGENT ")
print(agent)

# discretizer
dz = Discretizer(limits= limits, bins= bins)
dz.createBins()
print_msg_box(" BIN BOUNDARIES ")
pprint(dz.binBoundary)



# start game
p.init()
episode_idx = 1

while episode_idx < 100:
    if p.game_over():
        print(f"episode {episode_idx}")
        episode_idx += 1
        p.reset_game()

    gameState = p.getGameState()
    discreteState = dz.discretize(gameState)
    stateIdx = getGameStateIdx(discreteState)

    action = getActionFromIdx(agent.pickAction(stateIdx))
    reward = p.act(action)

    observation = p.getScreenRGB()

    



