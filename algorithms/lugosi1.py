import numpy as np
import matplotlib.pyplot as plt

import collections
import os

import logging
import pickle
logger = logging.getLogger(__name__)

from tqdm import tqdm
from .algorithm import *
from .utils import *

class Lugosi1(Algorithm):
    def __init__(self, environment):
        super().__init__(environment=environment)
        self.g = 128*self.env.K*np.log(3*self.env.K*self.env.M**2*self.env.T**2)
        #logger.info("g = ",self.g)
        self.proba_not_col = (1-1/self.env.K)**(self.env.M-1)
        #logger.info("pro not col", self.proba_not_col)

    def __str__(self):
        return f"Lugosi-Mehrabian 1"

    def reset(self):
        super().reset()
        self.phase = np.ones((self.env.M,))
        self.tau = np.zeros((self.env.M,))
        self.fixed = -1*np.ones((self.env.M,))
        self.fixed_ordered_arms = -1 * np.ones((self.env.M, self.env.K))

    def musical_chairs_single(self, player):
        if self.fixed[player] == -1:
            i = np.random.choice(self.env.M)
            arm = self.fixed_ordered_arms[player,i]
        else:
            arm = self.fixed[player]
        return int(arm)


    def run(self):
        arms_t = np.ones((self.env.M,)) * -1
        while self.t < self.env.T:
            players_phase_1 = np.where(self.phase == 1)[0]
            players_phase_2 = np.where(self.phase == 2)[0]
            players_phase_3 = np.where(self.phase == 3)[0]

            # phase 1 players
            arms_t[players_phase_1] = np.random.choice(np.arange(self.env.K),
                                     size=(len(players_phase_1),))
            self.tau[players_phase_1] += 1

            # phase 2 players
            arms_t[players_phase_2] = np.random.choice(np.arange(self.env.K),
                                     size=(len(players_phase_2),))
            self.tau[players_phase_2] -= 1

            # phase 3 players
            for player in players_phase_3:
                if self.tau[player] >= 1 and rewards_t[player] > 0:
                    self.fixed[player] = arms_t[player]
                arms_t[player] = self.musical_chairs_single(player)
                self.tau[player] += 1



            self.ordered_arms = np.argsort(-self.mu_hat, axis=1) # decreasing
            self.ordered_mu_hat = -np.sort(-self.mu_hat, axis=1)
            #logger.info("ordered arms", self.ordered_arms)
            #logger.info("mu_hat", self.mu_hat)


            # Passage phase 2
            #condition = (self.mu_hat[self.ordered_arms[:,self.env.M-1]] - \
            #             self.mu_hat[self.ordered_arms[:,self.env.M]]) \
            #            / self.proba_not_col >= 3*np.sqrt(self.g/self.tau)
            condition = ((self.ordered_mu_hat[:,self.env.M-1] - \
                         self.ordered_mu_hat[:,self.env.M]) \
                        / self.proba_not_col) >= 3*np.sqrt(self.g/self.tau)
            if self.t >= 8000 and self.t%3000==0 and False:
                logger.info("qté ", ((self.ordered_mu_hat[:,self.env.M-1] - \
                         self.ordered_mu_hat[:,self.env.M]) \
                        / self.proba_not_col))
                logger.info("3gtau", 3*np.sqrt(self.g/self.tau))
                logger.info("ordered_muhat", self.ordered_mu_hat)
                logger.info("mormu", self.ordered_mu_hat[:,self.env.M-1])
            #logger.info(self.ordered_mu_hat[:,self.env.M-1])
            #logger.info(self.ordered_mu_hat)
            for player in players_phase_1:
                if condition[player]:
                    #logger.info(f'***{player} passe 2')
                    self.phase[player] = 2
                    #logger.info(self.phase)
                    self.tau[player] *= 24
                    self.fixed_ordered_arms[player,:] = self.ordered_arms[player,:]

            # Passage phase 3
            for player in players_phase_2:
                if self.tau[player] == 0:
                   # logger.info(f'***{player} passe 3')
                    self.phase[player] = 3
                    #logger.info(self.phase)

            # Pull arms
            arms_t = arms_t.astype(int)
            rewards_t, regret_t, collisions_t = self.env.draw(arms_t)
            self.update_stats(arms_t=arms_t, rewards=rewards_t, regret_t=regret_t, collisions_t=collisions_t)
            self.t += 1
