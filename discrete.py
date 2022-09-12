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
        for key in self.limits.keys():
            self.binBoundary[key] = np.linspace(self.limits[key][0] ,self.limits[key][1], self.bins[key])
    

    def getBinIdx (self, key, val):
        """Return bin index given the value of a "key" variable"""
        if self.bins[key] == 1:     # ignore scenario
            return 0
        
        if val < self.limits[key][0]:   # invalid
            raise ValueError(f"Invalid value {val} less than {self.limits[key][0]}")
        
        if val > self.limits[key][1]:   # invalid
            raise ValueError(f"Invalid value {val} more than {self.limits[key][1]}")

        if val == self.limits[key][1]: # if upper limit, consider in last bin
            return (self.bins[key] - 1)
        

        # bin index starts from 0
        bin_idx = np.digitize(val, bins= self.binBoundary[key]) - 1
        
        return int(bin_idx)




    def discretize (self, gameState):

        return_dict = {}
        for key in gameState.keys():
            return_dict[key] = self.getBinIdx(key, gameState[key])

        return return_dict

    