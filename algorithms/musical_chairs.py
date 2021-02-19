import numpy as np
import matplotlib.pyplot as plt

import collections
import os

import logging
import pickle
logger = logging.getLogger()


from tqdm import tqdm
from .algorithm import *
from .utils import *

class MusicalChairs(Algorithm):
    def __init__(self, environment, T0):
        super().__init__(environment=environment)
        self.T0 = T0
        assert T0 < self.env.T

    def __str__(self):
        return f"MC sensing"

    def musical_chairs(self):
        arms_t = np.zeros((self.env.M,))
        for player in range(self.env.M):
            if self.fixed[player] == -1:
                if self.N_star[player] == 0:
                    i = 0
                else:
                    i = np.random.choice(self.N_star[player]) #0, 1, .. N_star-1
                arms_t[player] =  self.ordered_arms[player][self.env.K - i - 1]

            else:
                arms_t[player] = self.fixed[player]

        return arms_t.astype(int)

    def run(self):
        while self.t < self.T0: # explore
            arms_t = np.random.choice(np.arange(self.env.K),
                                     size=(self.env.M,))
            rewards_t, regret_t, collisions_t = self.env.draw(arms_t)
            self.update_stats(arms_t=arms_t, rewards=rewards_t, regret_t=regret_t, collisions_t=collisions_t)

            self.t += 1

        collision_per_player = np.sum(self.collisions, axis=1)

        self.N_star = np.minimum((np.log((self.T0-collision_per_player)/self.T0)/
                               np.log(1-1/self.env.K) + 1),
                           self.env.K).astype(int)
        #print("nustar", self.N_star)
        self.N_star[collision_per_player == self.T0] = self.env.K
        #print("nustar", self.N_star)
        self.ordered_arms = np.argsort(self.mu_hat, axis=1)
        #print(self.mu_hat)
        #print("ordered arms", self.ordered_arms)
        self.fixed = -1*np.ones((self.env.M,))
        while self.t < self.env.T:
            arms_t = self.musical_chairs()
            #print("chosen mc", arms_t)
            rewards_t, regret_t, collisions_t = self.env.draw(arms_t)
            non_collided_players = np.where(np.sum(collisions_t, axis=1) == 0)
            self.fixed[non_collided_players] = arms_t[non_collided_players]
            #print("fixed,",self.fixed)
            self.update_stats(arms_t=arms_t, rewards=rewards_t, regret_t=regret_t, collisions_t=collisions_t)
            self.env.update(t=self.t)
            self.t += 1
