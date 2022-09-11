
def print_msg_box(msg, indent=1, width=None, title=None):
    """Print message-box with optional title."""
    lines = msg.split('\n')
    space = " " * indent
    if not width:
        width = max(map(len, lines))
    box = f'╔{"═" * (width + indent * 2)}╗\n'  # upper_border
    if title:
        box += f'║{space}{title:<{width}}{space}║\n'  # title
        box += f'║{space}{"-" * len(title):<{width}}{space}║\n'  # underscore
    box += ''.join([f'║{space}{line:<{width}}{space}║\n' for line in lines])
    box += f'╚{"═" * (width + indent * 2)}╝'  # lower_border
    print(box)

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

def getGameStateIdx (discrete_gameState, binNums):
    """Return idx of discretized gameState, idx belongs in range 0 to NUM_STATE"""
    idx = 0
    b = 1
    # get bins for every state variable
    for key in discrete_gameState.keys():
        idx = discrete_gameState[key] + b*idx
        b = binNums[key]
    
    return idx