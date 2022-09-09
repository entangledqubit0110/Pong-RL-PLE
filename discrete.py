import numpy as np

class Discretizer:
    def __init__(self, bins, limits):
        # maximum and minimum for variables
        # dict contains a tuple of (min, max) for eachs
        self.limits = limits

        # number of bins for discretization
        # a dict that contains no of bins for each
        self.bins = bins

    def createBins (self):
        """Create bins using numpy and given limits and number of bins"""
        self.binBoundary = {}
        self.binBoundary["player_y"] = np.linspace(self.limits['player_y'][0] ,self.limits['player_y'][1], self.bins['player_y'])
        self.binBoundary["player_velocity"]: np.linspace(self.limits["player_velocity"][0], self.limits["player_velocity"][1], self.bins["player_velocity"])
        self.binBoundary["cpu_y"] = np.linspace(self.limits["cpu_y"][0] ,self.limits["cpu_y"][1], self.bins["cpu_y"])
        self.binBoundary["ball_x"] = np.linspace(self.limits["ball_x"][0] ,self.limits["ball_x"][1], self.bins["ball_x"])
        self.binBoundary["ball_y"] = np.linspace(self.limits["ball_y"][0] ,self.limits["ball_y"][1], self.bins["ball_y"])
        self.binBoundary["ball_velocity_x"] = np.linspace(self.limits["ball_velocity_x"][0], self.limits["ball_velocity_x"][1], self.bins["ball_velocity_x"])
        self.binBoundary["ball_velocity_y"] = np.linspace(self.limits["ball_velocity_y"][0], self.limits["ball_velocity_y"][1], self.bins["ball_velocity_y"])

    def getBinIdx (self, key, val):
        """Return bin index given the value of a "key" variable"""
        if val < self.limits[key][0]:   # invalid
            raise ValueError(f"Invalid value {val} less than {self.limits[key][0]}")
        
        if val > self.limits[key][1]:   # invalid
            raise ValueError(f"Invalid value {val} more than {self.limits[key][1]}")

        if val == self.limits[key][1]: # if upper limit, consider in last bin
            return (self.bins[key] - 1)
        
        if self.bins[key] == 1:     # ignore scenario
            return 0

        bin_idx = np.digitize(val, bins= self.binBoundary[key]) - 1
        
        return bin_idx




    def discretize (self, gameState):

        return_dict = {}
        for key in gameState.keys():
            return_dict[key] = self.getBinIdx(key, gameState[key])

        return return_dict

    