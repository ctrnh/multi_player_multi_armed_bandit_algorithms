import numpy as np
import matplotlib.pyplot as plt

import collections
import os

import logging
import pickle
logger = logging.getLogger()
logger.setLevel(logging.INFO)

import time
from tqdm import tqdm
from .algorithm import *
from .utils import *
try:
    from cklucb import computeKLUCB
except:
    pass

class Dynmc(Algorithm):
    """
    Dynamic Musical Chairs
    """
    def __init__(self, environment, ):
        super().__init__(environment=environment)



    def reset(self):
        super().reset()
        self.idx_hist = []
        self.gamma_hist = []
        self.fixed = -1*np.ones(self.env.M)
        if self.env.K == 2 and  self.save_optional:
            self.idx_hist = np.zeros((self.env.M, self.env.K, self.env.T))
            self.gamma_hist = np.zeros((self.env.M, self.env.T))

    def __str__(self):
        return 'DYN-MC'

    def reset_player(self, leaving_players):
        #super().reset_player(leaving_players)
        for player in leaving_players:
            self.fixed[player] = -1

    def run(self):
        self.reset()
        tic = time.time()
        while self.t < self.env.T:
            if self.t==100000:
                time.strftime("%Hh%M", time.localtime())
                print(f"Estimated time remaining: {round(((self.env.T-1e5)*(time.time()-tic)*1e-5)/60,1)} min")

            arms_t = -1*np.ones(self.env.M)
            for player in self.env.active_players:
                if self.fixed[player] != -1:
                    arms_t[player] = self.fixed[player]
                else:
                    arms_t[player] = np.random.randint(self.env.K)

            arms_t = arms_t.astype(int)
            rewards_t, regret_t, collisions_t, system_reward_t = self.env.draw(arms_t,)

            for player in self.env.active_players:
                if self.fixed[player] == -1 and rewards_t[player] == 1:
                    self.fixed[player] = arms_t[player]


            self.update_stats(arms_t=arms_t,
                                rewards=rewards_t,
                                regret_t=regret_t,
                                collisions_t=collisions_t,
                                system_reward=system_reward_t
                                )
            leaving_players = self.env.update(self.t)
            self.reset_player(leaving_players)
            self.t += 1
