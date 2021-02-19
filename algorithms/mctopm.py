import numpy as np
import matplotlib.pyplot as plt

import collections
import os

import logging
import pickle
logger = logging.getLogger()
logger.setLevel(logging.INFO)

from tqdm import tqdm
from .algorithm import *
from .utils import *

logger = logging.getLogger()
logger.setLevel(logging.INFO)


cythonized_KLUCB = True
try:
    from cklucb import computeKLUCB
except:
    cythonized_KLUCB = False
    pass

class MCTopM(Algorithm):
    def __init__(self, environment, ucb_type="KLUCB"):
        super().__init__(environment=environment)
        self.sensing = True
        self.ucb_type = ucb_type
        self.klucb = np.vectorize(klucb_ber)
        self.c = 0


    def reset(self):
        super().reset()
        self.idx_hist = np.zeros((self.env.M, self.env.K, self.env.T))
        self.s = np.zeros((self.env.M,))
    def __str__(self):
        return f"MCTopM"

    def compute_ucb_idx(self):
        ucb_idx = np.zeros((self.env.M,
                            self.env.K))
        if self.ucb_type == "UCB1":
            ucb_idx = self.mu_hat + np.sqrt((2*np.log(self.t))/self.pulls)
        elif self.ucb_type == "KLUCB":
            if cythonized_KLUCB:
                N_t =  (self.pulls-self.collisions).astype(np.int32)
                ucb_idx = computeKLUCB(self.t, self.mu_hat,N_t)
            else:
                d = (np.log(self.t) + self.c*np.log(np.log(self.t))) / (self.pulls+1e-7)
                ucb_idx = self.klucb(self.mu_hat, d)
        else:
            raise ValueError
        ucb_idx[self.pulls - self.collisions == 0] = float('inf')
       # ucb_idx = np.minimum(ucb_idx,1)
        return ucb_idx

    def compute_M_hat(self, ucb_idx):
        #M_hat_t = np.argpartition(-new_ucb_idx, self.env.M-1, axis=1)[:,:self.env.M] # MxM
        arm_idx_sorted = np.argsort(-ucb_idx)
        M_hat = [[] for player in range(self.env.M)]

        for player in range(self.env.M):
            ucb_pivot = ucb_idx[player, arm_idx_sorted[player,self.env.M-1]]
            for arm in (arm_idx_sorted[player,:]):
                if ucb_idx[player, arm] >= ucb_pivot:
                    M_hat[player].append(arm)
                else:
                    break

        return M_hat

    def run(self):
        arms_t = np.random.choice(np.arange(self.env.K),
                                     size=(self.env.M,))
        #collisions_t = np.zeros((self.env.M, self.env.K))

        ucb_idx = self.compute_ucb_idx()

        rewards_t, regret_t ,collisions_t = self.env.draw(arms_t,sensing=self.sensing)
        #logger.info(f"drawing {arms_t}, regret:{regret_t}")
        self.update_stats(arms_t=arms_t, rewards=rewards_t, regret_t=regret_t, collisions_t=collisions_t)
        self.env.update(t=self.t)
        self.t += 1
        #logger.info(f"after update, mu_hat = \n {self.mu_hat}\n")
        while self.t < self.env.T:
            new_ucb_idx = self.compute_ucb_idx()
            #self.idx_hist[:,:,self.t] = new_ucb_idx # MxK


            M_hat_t = self.compute_M_hat(ucb_idx=new_ucb_idx)
            #if self.t%1==0:
                #logger.info(f"\n.......t = {self.t},")
                #logger.info("ucb_idx",new_ucb_idx)
                #logger.info("M_hat_t",M_hat_t)
            new_arms_t = np.zeros((self.env.M,))
            for player in range(self.env.M):
                if arms_t[player] not in M_hat_t[player]:
                    new_set = np.intersect1d(M_hat_t[player],
                                            np.nonzero(ucb_idx[player,:] <= ucb_idx[player, arms_t[player]]))

                    new_arms_t[player] = np.random.choice(new_set)
                    #assert ucb_idx[player, int(new_arms_t[player])] <= ucb_idx[player,int(arms_t[player])]
                    #logger.info(f"t = {self.t}: player {player}, arms_t = {arms_t[player]} not in M_hat_t={M_hat_t[player]}. So pull {new_arms_t[player]} from set {new_set}")
                    self.s[player] = 0
                elif np.sum(collisions_t[player,:]) >= 1 and not self.s[player]:
                    new_arms_t[player] = np.random.choice(M_hat_t[player])
                    self.s[player] = 0
                    #logger.info(f"t = {self.t}: player {player}, arms_t = {arms_t[player]} collided. So pull {new_arms_t[player]} from M_hat_t={M_hat_t[player]} ")
                else:
                    new_arms_t[player] = arms_t[player]
                    #if self.s[player] != 1:
                        #logger.info(f"t = {self.t}: player {player}, arms_t = {arms_t[player]} is in M_hat_t and did not collide -> fixing ")
                    self.s[player] = 1

            arms_t = new_arms_t.astype(int)
            ucb_idx = new_ucb_idx

            rewards_t, regret_t ,collisions_t = self.env.draw(arms_t, sensing=self.sensing)

            self.update_stats(arms_t=arms_t, rewards=rewards_t, regret_t=regret_t, collisions_t=collisions_t)
            self.env.update(t=self.t)
            self.t += 1
            #if self.t%1==0:
                ##logger.info(f"Now successes: {self.successes}")
                ##logger.info(f"Now pulls: {self.pulls}")
                ##logger.info(f"Now collisions: {self.collisions}")
                #logger.info(f"t = {self.t}: drawing {arms_t}, reward: {rewards_t}, regret:{regret_t}")
                #logger.info(f"Now mu_hat: \n{self.mu_hat}\n, successes: \n {self.successes}, pulls:\n{self.pulls}\n")
