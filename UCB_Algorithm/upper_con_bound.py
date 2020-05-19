''' 
This algorithm is based on the Upper Confidence bound reinforcement learning
algorithm 

Inputs Parameters and Usage

The only input is the dataframe you require
ubc = upper_con_bound.UBC()
results = u.learn(data)

'data' should be a numpy.ndarray

***********FINAL VERSION ***********
'''

import numpy as np
import math

class UBC:

    def __init__(self, x = 0):
        self.x = x
    
    def learn(self,data):
        c = len(data[0])
        N = len(data)
        results = []
        N_n = [0]*c
        R_n = [0]*c

        for n in range(0,N):
            max_ucb = 0
            ad = 0

            for i in range(0,c):
                if N_n[i]>0:
                    #UBC calculation
                    average_r = R_n[i] / N_n[i]
                    delta_i = math.sqrt(3/2 * math.log(n + 1) / N_n[i])
                    ucb = average_r + delta_i
               
                else:
                    #Allocating a Large value at first round
                    ucb = 1e500 

                if ucb>max_ucb: 
                    max_ucb = ucb
                    ad = i
                
            results.append(ad) 
            N_n[ad] = N_n[ad] + 1 
            reward = data[n][ad] 
            R_n[ad] = R_n[ad] + reward 
        
        return results




