''' 
This algorithm is based on the Thompson Sampling reinforcement learning
algorithm 

Inputs Parameters and Usage

The only input is the dataframe you require
ts = thompson.TS()
results = ts.learn(data)

'data' should be a numpy.ndarray

***********FINAL VERSION ***********
'''

import numpy as np
import math
import random

class TS:

    def __init__(self, x = 0):
        self.x = x
    


    def learn(self, data):
        
        c = len(data[0])
        N = len(data)
        N0_n = [0]*c
        N1_n = [0]*c
        results = []

        for n in range(0,N):
            p = 0
            max_beta = 0
            for i in range(0,c):
                ran_beta = random.betavariate(N1_n[i]+1,N0_n[i]+1)

                if ran_beta > max_beta:
                    max_beta = ran_beta
                    p = i
            results.append(p)
            reward = data[n][p] 
            
            if reward==1:
                N1_n[p] = N1_n[p] + 1
            else:
                N0_n[p] = N0_n[p] + 1
        return results