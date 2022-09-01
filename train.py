from ple.games.pong import Pong
from ple import PLE
from discrete import Discretizer
from agents.monte_carlo import MonteCarlo
import pprint

# game panel details
WIDTH = 220
HEIGHT = 148
FPS = 50

# discretization params
NUM_POS_BINS = 8
NUM_BALL_X_BINS = NUM_POS_BINS
NUM_BALL_Y_BINS = NUM_POS_BINS
NUM_PLAYER_Y_BINS = NUM_POS_BINS
NUM_CPU_Y_BINS = NUM_POS_BINS

NUM_VEL_BINS = 6
NUM_PLAYER_VEL_BINS = NUM_VEL_BINS
NUM_BALL_X_VEL_BINS = NUM_VEL_BINS
NUM_BALL_Y_VEL_BINS = NUM_VEL_BINS

NUM_STATES =    (NUM_BALL_X_BINS*
                NUM_BALL_Y_BINS*
                NUM_PLAYER_Y_BINS*
                NUM_CPU_Y_BINS*
                NUM_BALL_X_VEL_BINS*
                NUM_BALL_Y_VEL_BINS*
                NUM_PLAYER_VEL_BINS)

NUM_ACTIONS = 3

# utility functions
def getGameStateIdx (gameState):
    """Return idx of discretized gameState, idx belongs in range 0 to NUM_STATE"""
    # get bins for every state variable
    player_y = gameState["player_y"]
    player_velocity_y = gameState["player_velocity"]
    cpu_y = gameState["cpu_y"]
    ball_x = gameState["ball_x"]
    ball_y = gameState["ball_y"]
    ball_velocity_x = gameState["ball_velocity_x"]
    ball_velocity_y = gameState["ball_velocity_y"]

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

def getActionIdx (action):
    """Get idx for 3 types of actions: 115, 119 and None"""
    if action is None:
        return 0
    elif action == 115:
        return 1
    elif action == 119:
        return 2

def getActionFromIdx (idx):
    """Get action from idx"""
    if idx == 0:
        return None
    elif idx == 1:
        return  115
    elif idx == 2:
        return 119


# initialize game
game = Pong(width= WIDTH, height= HEIGHT)
p = PLE(game, fps=FPS, display_screen=True, force_fps=True)


agent = MonteCarlo(num_states= NUM_STATES, num_actions= NUM_ACTIONS)

# discretizer
dz = Discretizer(game, num_pos_bins=NUM_POS_BINS, num_velocity_bins=NUM_VEL_BINS)





episode_idx = 0
generation_idx = 0
episode_in_generation = 10

while True:
    # start game
    p.init()

    # stores episodes for one generation
    episodes = []
    
    # store for every episode
    states = []
    actions = []
    rewards = []
    k = 0
    while True: 
        if p.game_over():
            # store episode information
            episode_idx += 1        
            episode = [states, actions, rewards]
            print(actions)
            episodes.append(episode)
            print(f"episode {episode_idx}: {sum(rewards)}")

            k += 1
            # check if end of an generation
            if k == episode_in_generation:
                generation_idx += 1
                print(f"END OF generation {generation_idx}")
                break
            
            # reset before next episode
            states.clear()
            actions.clear()
            rewards.clear()
            p.reset_game()

        observation = p.getScreenRGB()
        
        # state
        gameState = game.getGameState()
        discreteState = dz.discretize(gameState)
        stateIdx = getGameStateIdx(discreteState)
        
        # logging
        # print(gameState)
        # print(discreteState)
        # print(stateIdx)

        states.append(stateIdx)

        # action
        actionIdx = agent.pickAction(stateIdx)
        action = getActionFromIdx(actionIdx)

        # print(action, type(action))
        # print(actionIdx)
        
        actions.append(actionIdx)

        # reward
        reward = p.act(action)
        
        # print(reward)

        rewards.append(reward)

    agent.update(episodes=episodes)