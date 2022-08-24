class Discretizer:
    def __init__(self, game, num_pos_bins, num_velocity_bins):
        # limits for position
        self.MIN_Y_POS = 0
        self.MAX_Y_POS = game.height
        self.MIN_X_POS = 0
        self.MAX_X_POS = game.width
        # limits for velocity
        self.MAX_BALL_X_VELOCITY = game.ball.speed
        self.MAX_BALL_Y_VELOCITY = game.ball.speed
        self.MIN_BALL_X_VELOCITY = -1*game.ball.speed
        self.MIN_BALL_Y_VELOCITY = -1*game.ball.speed
        self.MAX_PLAYER_VELOCITY = game.agentPlayer.speed
        self.MIN_PLAYER_VELOCITY = -1*game.agentPlayer.speed

        print(self.MAX_BALL_X_VELOCITY, self.MIN_BALL_X_VELOCITY)
        print(self.MAX_BALL_Y_VELOCITY, self.MIN_BALL_Y_VELOCITY)

        # number of bins for discretization
        self.num_pos_bins = num_pos_bins
        self.num_velocity_bins = num_velocity_bins

    def getBin (self, val, min, max, num_bins):
        """Return bin index given the value of a variable
        and its min & max as well as intended number of bins"""
        if val < min:   # invalid
            print(f"Invalid value {val} less than {min}")
            return -1
        
        if val > max:   # invalid 
            print(f"Invalid value {val} more than {max}")
            return -1

        bin_idx = 0                     # idx of the curr bin
        delta = (max - min)/num_bins    # length of a bin
        temp = min + delta              # the upper limit of current bin
        while temp <= max:
            if val <= temp:  # falls in curr bin
                break
            else:
                temp += delta 
                bin_idx += 1
        
        return bin_idx




    def discretize (self, gameState):
        player_y = gameState["player_y"]
        player_velocity_y = gameState["player_velocity"]
        cpu_y = gameState["cpu_y"]
        ball_x = gameState["ball_x"]
        ball_y = gameState["ball_y"]
        ball_velocity_x = gameState["ball_velocity_x"]
        ball_velocity_y = gameState["ball_velocity_y"]

        d_player_y = self.getBin(player_y, self.MIN_Y_POS ,self.MAX_Y_POS, self.num_pos_bins)
        d_player_vel_y = self.getBin(player_velocity_y, self.MIN_PLAYER_VELOCITY, self.MAX_PLAYER_VELOCITY, self.num_velocity_bins)

        d_cpu_y = self.getBin(cpu_y, self.MIN_Y_POS ,self.MAX_Y_POS, self.num_pos_bins)

        d_ball_x = self.getBin(ball_x, self.MIN_X_POS ,self.MAX_X_POS, self.num_pos_bins)
        d_ball_y = self.getBin(ball_y, self.MIN_Y_POS ,self.MAX_Y_POS, self.num_pos_bins)
        d_ball_vel_x = self.getBin(ball_velocity_x, self.MIN_BALL_X_VELOCITY, self.MAX_BALL_X_VELOCITY, self.num_velocity_bins)
        d_ball_vel_y = self.getBin(ball_velocity_y, self.MIN_BALL_Y_VELOCITY, self.MAX_BALL_Y_VELOCITY, self.num_velocity_bins)
        
        return_dict =   {
                            "player_y": d_player_y,
                            "player_velocity": d_player_vel_y,
                            "cpu_y": d_cpu_y,
                            "ball_x": d_ball_x,
                            "ball_y": d_ball_y,
                            "ball_velocity_x": d_ball_vel_x,
                            "ball_velocity_y": d_ball_vel_y
                        }
        return return_dict

    