from ple.games.pong import Pong
from ple import PLE
import random
from discrete import Discretizer

WIDTH = 220
HEIGHT = 148
FPS = 30
NUM_POS_BINS = 2
NUM_VEL_BINS = 10

# initialize game
game = Pong(width= WIDTH, height= HEIGHT)
p = PLE(game, fps=FPS, display_screen=True, force_fps=False)


# placeholder agent
class myAgentHere:
    def __init__(self, allowed_actions):
        self.actions = allowed_actions

    def pickAction(self, r, o):
        return random.choice(self.actions)

# initialize placeholder agent
agent = myAgentHere(allowed_actions=p.getActionSet())


# discretizer
dz = Discretizer(game, num_pos_bins=NUM_POS_BINS, num_velocity_bins=NUM_VEL_BINS)




p.init()
reward = 0.0
nb_frames = 1000
for i in range(nb_frames):
    if p.game_over():
            p.reset_game()

    observation = p.getScreenRGB()
    
    gameState = game.getGameState()
    print(gameState)

    print(dz.discretize(gameState))

    action = agent.pickAction(reward, observation)
    reward = p.act(action)