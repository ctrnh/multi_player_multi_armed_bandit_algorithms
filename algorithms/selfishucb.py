import numpy as np
import matplotlib.pyplot as plt

import collections
import os

import logging
import pickle
logger = logging.getLogger(__name__)

import time
from tqdm import tqdm
from .algorithm import *
from .utils import *

cythonized_KLUCB = True
try:
    from cklucb import computeKLUCB
except:
    cythonized_KLUCB = False
    pass

class Selfishucb(Algorithm):
    """
    SelfishUCB algorithm (randomized or not)
    """
    def __init__(self, environment, randomized, ucb_type="KLUCB"):
        super().__init__(environment=environment)
        self.ucb_type = ucb_type
        self.klucb = np.vectorize(klucb_ber)
        self.c = 3
        self.randomized = randomized
        if not cythonized_KLUCB:
            logger.info("Failed to import cythonized KLUCB. Running may be slower. (To use cythonized version you must compile the cythonized version in the cklucb folder)")

    def reset(self):
        super().reset()
        self.idx_hist = []
        self.gamma_hist = []

        if self.env.K == 2 and  self.save_optional:
            self.idx_hist = np.zeros((self.env.M, self.env.K, self.env.T))
            self.gamma_hist = np.zeros((self.env.M, self.env.T))

    def __str__(self):
        name = f"Selfish{self.ucb_type}"
        if self.randomized:
            return "Rnd-" + name
        return name



    def compute_ucb_idx(self):
        ucb_idx = np.zeros((self.env.M,
                            self.env.K))

        if cythonized_KLUCB:
            ucb_idx = computeKLUCB(self.t, self.mu_hat, self.pulls)
        else:
            d = (np.log(self.t) + self.c*np.log(np.log(self.t))) / (self.pulls+1e-7)
            ucb_idx = self.klucb(self.mu_hat, d)
        ucb_idx[self.pulls < 1] = float('+inf')
        return ucb_idx



    def run(self):
        self.reset()
        tic = time.time()
        logger.info(f"Running {str(self)}... for T = {self.env.T} steps")
        while self.t < self.env.T:
            if self.t==100000:
                time.strftime("%Hh%M", time.localtime())
                logger.info(f"Estimated time remaining: {round(((self.env.T-1e5)*(time.time()-tic)*1e-5)/60,1)} min")
            ucb_idx = self.compute_ucb_idx()

            if self.randomized:
                ucb_idx +=  np.random.normal(0, 1/(self.t+1), size=(self.env.M, self.env.K))


            arms_t = best_arm(ucb_idx)



            if not self.env.dynamic:
                rewards_t, regret_t, collisions_t = self.env.draw(arms_t,)

                self.update_stats(arms_t=arms_t,
                                    rewards=rewards_t,
                                    regret_t=regret_t,
                                    collisions_t=collisions_t)
                self.t += 1
            else:
                rewards_t, regret_t, collisions_t, system_reward_t = self.env.draw(arms_t,)
                self.update_stats(arms_t=arms_t,
                                    rewards=rewards_t,
                                    regret_t=regret_t,
                                    collisions_t=collisions_t,
                                    system_reward=system_reward_t
                                    )
                leaving_players = self.env.update(self.t)
                self.reset_player(leaving_players)
                self.t += 1
