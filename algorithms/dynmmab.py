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
EPS = 1e-5

class Dynmmab(Algorithm):
    def __init__(self, environment,
               ):
        super().__init__(environment=environment)

        self.t_entries = np.array(self.env.t_entries[:self.env.M])
        ##logger.info(self.env.individual_horizons)
    def __str__(self):
        return f"DYN-MMAB"

    def reset(self):
        super().reset()
        self.fixed = -1 * np.ones(self.env.M)
        self.preferences = [[] for j in range(self.env.M)]
        self.occupied = [set() for j in range(self.env.M)]
        self.active_arms = [{k for k in range(self.env.K)} for j in range(self.env.M)]

        self.p = np.zeros((self.env.M,), dtype=int)
        # self.r_inf = np.zeros((self.env.M, self.env.K))
        # self.r_sup = np.ones((self.env.M, self.env.K))
        self.L = np.inf * np.ones((self.env.M, self.env.K))

        self.T_temp = np.zeros((self.env.M, self.env.K))
        self.S_temp = np.zeros((self.env.M, self.env.K))

    def run(self):
        assert environment.dynamic == True
        while self.t <= self.env.T:
            list_env_active_players = list(self.env.active_players)

            #### Explore 1 step
            arms_t = -1 * np.ones((self.env.M,))
            for player in list_env_active_players:
                if self.fixed[player] != -1:
                    arms_t[player] = self.fixed[player]
                else:
                    arms_t[player] = np.random.choice(self.env.K)

            arms_t = arms_t.astype(int)

            rewards_t, regret_t, collisions_t, system_reward_t = self.env.draw(arms_t)
            self.update_stats(arms_t=arms_t, rewards=rewards_t, regret_t=regret_t, collisions_t=collisions_t, system_reward=system_reward_t)
            _ = self.env.update(self.t)
            self.t += 1

            #### Update
            self.T_temp[list_env_active_players, arms_t[list_env_active_players]] += 1
            self.S_temp[list_env_active_players, arms_t[list_env_active_players]] += rewards_t[list_env_active_players]
            ##logger.info("T_temp,", self.T_temp)
            ##logger.info(self.t, self.t_entries)
            B = 2*np.sqrt((6*self.env.K*np.log(self.env.individual_horizons))/(np.maximum(0,self.t - self.t_entries) + EPS))

            for player in list_env_active_players:
                if self.fixed[player] == -1:
                    k = arms_t[player]

                    self.L[player, k] = min(self.L[player, k], (2*np.exp(1)*np.log(self.env.individual_horizons[player]))/(max(EPS,self.mu_hat[player,k] - B[player])))

                    if len(self.preferences[player]) > 0:
                        if k == self.preferences[player][self.p[player]] and rewards_t[player] > 0:
                            self.fixed[player] = k
                            #logger.info(f"... t = {self.t}: player", player, "is now fixed on ",k)
                            self.preferences[player] = []
                            if k in self.active_arms[player]:
                                self.active_arms[player].remove(k)

                        elif self.preferences[player][self.p[player]] in self.occupied[player]:
                            self.p[player] += 1


                    if self.fixed[player] == -1:
                        ##  bcp de 0 sur le bras k => occupé
                        if self.T_temp[player, k] >= self.L[player, k] and k not in self.occupied[player]:
                            if self.S_temp[player, k] == 0:
                                self.occupied[player].add(k)
                                if k in self.active_arms[player]:
                                    self.active_arms[player].remove(k)
                                #logger.info(f"... t = {self.t}: player {player} puts active arm {k} to occupied because S_temp=0. \n Now, preferences:  {self.preferences}. and p: {self.p} and active_arms: {self.active_arms}, and occupied: {self.occupied}")
                            self.S_temp[player, k] = 0
                            self.T_temp[player, k] = 0

                        if len(self.active_arms[player]) >=1: # at least one active arm to place in preferences
                            ## an active arm is better than the other
                            list_active_arms_player = list(self.active_arms[player])
                            current_mu_best = -1
                            best_active_arm_player = -5
                            for arm_best in range(self.env.K):
                                if arm_best in self.active_arms[player]:
                                    if self.mu_hat[player, arm_best] > current_mu_best:
                                        best_active_arm_player = arm_best
                                        current_mu_best = self.mu_hat[player, arm_best]

                            add_best_to_preference = True
                            for arm_l in list(self.active_arms[player]):
                                if arm_l != best_active_arm_player:
                                    #lb of best arm < ub of an active arm
                                    if max(0, self.mu_hat[player, best_active_arm_player] - B[player]) < min(1, self.mu_hat[player,arm_l] + B[player]):
                                
                                        add_best_to_preference = False
                                        break
                            if add_best_to_preference :
                                #logger.info(f" t = {self.t} player {player}")
                                self.preferences[player].append(best_active_arm_player)
                                if best_active_arm_player in self.active_arms[player]:
                                    self.active_arms[player].remove(best_active_arm_player)

                                #logger.info(f"... t = {self.t}: player {player} puts active arm {best_active_arm_player} to preferences. Now, preferences:  {self.preferences}. and p: {self.p} and active_arms: {self.active_arms}")




                        if len(self.preferences[player]) > 0:
                            ## an arm becomes less good => means that it is now occupied
                            for arm_l in range(self.env.K):
                                if arm_l not in self.preferences[player][:self.p[player]] and arm_l in self.active_arms[player]:
                                    if max(0, self.mu_hat[player,arm_l] - B[player]) > min(1,self.mu_hat[player, self.preferences[player][self.p[player]]]):
                                        #logger.info(f"... t = {self.t}: player {player} puts {self.preferences[player][self.p[player]]} to occupied because there exists arm_l not in the p first preferred with r_inf[arm_l] > r_sup[{self.preferences[player][self.p[player]]}]")
                                        self.occupied[player].add(self.preferences[player][self.p[player]])
                                        if self.preferences[player][self.p[player]] in self.active_arms[player]:
                                            self.active_arms[player].remove(self.preferences[player][self.p[player]])
