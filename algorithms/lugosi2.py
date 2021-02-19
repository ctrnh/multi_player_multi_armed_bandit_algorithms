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

class Lugosi2(Algorithm):
    def __init__(self, environment, mu_lb=None ):
        super().__init__(environment=environment)

        self.mu_lb=mu_lb
        if self.mu_lb is None:
            self.mu_lb = max(self.env.mu[self.env.M-1]-0.05, 0.01)
        ##logger.info("mu_lb =", self.mu_lb)
        self.proba_not_col = (1-1/self.env.K)**(self.env.M-1)

        self.g_2 = np.log(4*self.env.M**3*self.env.T**2*self.env.K)/2
        ordered_mu = -np.sort(-np.array(self.env.mu)
                             )
        self.alpha = 4*self.env.K*np.log(6*self.env.K*self.env.M**2*self.env.T)/self.mu_lb
        ##logger.info(ordered_mu[self.env.M-1])
        self.alpha = int(self.alpha)
        ##logger.info("g = ", self.g_2, "alpha = " ,self.alpha)
    def __str__(self):
        return "Lugosi-Mehrabian 2"

    def reset(self):
        super().reset()
        self.phase = np.ones((self.env.M,))
        self.tau = np.zeros((self.env.M,))

        self.G = [[] for i in range(self.env.M)]
        self.B = [[] for i in range(self.env.M)]
        self.S = [[k for k in range(self.env.K)] for i in range(self.env.M)]
        self.fixed = -1*np.ones((self.env.M,))

    def add_to(self, arm, player, which):
        self.S[player].remove(arm)


    def mc_alpha(self, set_arms, fixing_golden=False, fixed=None):
        """
        Performs MC for selt.alpha time step, with set of arms set_arms
        """
        if fixed is None:
            fixed = self.fixed.copy()

        for i_alpha in range(self.alpha):
            arms_t = np.ones((self.env.M,)) * -1
            for player in range(self.env.M):
                if self.fixed[player] != -1: # fixed in golden
                    arms_t[player] = self.fixed[player]
                elif fixed[player] != -1: # fixed in MC or fixed was not -1
                    arms_t[player] = fixed[player]
                else: # not fixed
                    arms_t[player] = np.random.choice(self.env.K)


            arms_t = arms_t.astype(int)
            rewards_t, regret_t, collisions_t = self.env.draw(arms_t)
            #if i_alpha <=3:
                #logger.info(f"........... MC alpha drawing {arms_t}, fixing golden : {fixing_golden} ")
            self.update_stats(arms_t=arms_t, rewards=rewards_t, regret_t=regret_t, collisions_t=collisions_t)
            self.t += 1
            for player in range(self.env.M):
                if rewards_t[player] > 0 and arms_t[player] in set_arms[player]:
                    fixed[player] = arms_t[player]
                    if fixing_golden:
                        self.fixed[player] = arms_t[player]
                        ##logger.info(f"----- t = {self.t}, user {player} now fixed on: {self.fixed}")
        return fixed

    def run(self):
        successes_orthogonalized = np.zeros((self.env.M, self.env.K))
        draws_orthogonalized = 1e-5*np.ones((self.env.M, self.env.K))
        mu_hat = np.zeros((self.env.M, self.env.K))

        epoch = 1
        while self.t < self.env.T:
            #logger.info(f"...........................\nNEW EPOCH : i ={epoch}\n....................")
            #logger.info(f"t = {self.t} golden set: {self.G}. Bad set: {self.B}")
            self.mc_alpha(self.G, fixing_golden=True)

            E = [[] for i in range(self.env.M)]
            for iteration in range(self.env.K + self.env.M -1): # explor silver
                S_no_E = [np.setdiff1d(self.S[player],E[player]) for player in range(self.env.M)]
                #logger.info(f"----- t = {self.t}, iteration = {iteration}, S not Explored: {S_no_E}")

                # For alpha rounds, either draw a unexplored silver arm, or a random arm in [K]
                js = self.mc_alpha(S_no_E)

                #logger.info(f"----- t = {self.t}, iteration = {iteration}, outcome of MC on S not Explored: {js}")
                # For alpha rounds, either draw a the previous explored silver arm, or a random silver arm
                js = self.mc_alpha(self.S, fixed=js)

                #logger.info(f"----- t = {self.t}, final arms to draw in this iteration: {js}")
                arms_t = js.copy()
                arms_t = arms_t.astype(int)
                for t_explore in range(2**epoch):
                    rewards_t, regret_t, collisions_t = self.env.draw(arms_t)
                    #if t_explore <=3:
                        #logger.info(f"........... Exploration drawing {arms_t}")
                    self.update_stats(arms_t=arms_t, rewards=rewards_t, regret_t=regret_t, collisions_t=collisions_t)
                    self.t += 1
                    for player in range(self.env.M):
                        successes_orthogonalized[player, arms_t[player]] += rewards_t[player]
                        draws_orthogonalized[player, arms_t[player]] += 1

                for player in range(self.env.M):
                    if js[player] not in E[player]:
                        E[player].append(js[player]) #js[player] is now explored by player
                mu_hat = successes_orthogonalized/draws_orthogonalized
                #logger.info(f"----- t = {self.t}, finished exploration for {2**epoch} steps, arms which have been explored: {arms_t}, \n----- current mu_hat \n{np.round(mu_hat,2)}, draws: \n{np.round(draws_orthogonalized,2)}")

            #logger.debug(f"\nEND EPOCH")

            for player in range(self.env.M): # unexplored arms are either G or B
                for arm in S_no_E[player]:
                    if mu_hat[player][arm] - np.sqrt(self.g_2/2**(epoch-1)) > self.mu_lb:
                        self.S[player].remove(arm)
                        assert arm not in self.G[player]
                        self.G[player].append(arm)
                        #logger.info(f"----- t = {self.t}, user {player}: unexplored silver arm {arm} is Golden because unexplored and {mu_hat[player][arm] - np.sqrt(self.g_2/2**(epoch-1))} > mu_lb = {self.mu_lb}")
                    else:
                    #elif mu_hat[player][arm] + np.sqrt(self.g_2/2**(epoch-1)) < self.mu_lb:
                        self.S[player].remove(arm)
                        assert arm not in self.B[player]
                        self.B[player].append(arm)
                        #logger.info(f"----- t = {self.t}, user {player}: unexplored silver arm {arm} is Bad because unexplored and {mu_hat[player][arm] - np.sqrt(self.g_2/2**(epoch-1))} < mu_lb = {self.mu_lb}")


                to_remove_from_S = []
                to_append_to_B = []
                to_append_to_G = []
                for arm in self.S[player]: # all explored silver
                    count_b = 0
                    count_g = 0
                    for arm_count in self.S[player]:
                        if mu_hat[player][arm_count] - np.sqrt(self.g_2/2**epoch) > mu_hat[player][arm] + np.sqrt(self.g_2/2**epoch):
                            count_b += 1
                        elif mu_hat[player][arm_count] + np.sqrt(self.g_2/2**epoch) < mu_hat[player][arm] - np.sqrt(self.g_2/2**epoch):
                            count_g += 1
                    ##logger.info(f"----- t = {self.t}, user {player}: arm {arm} (count_b = {count_b}) S[player] = {self.S[player]}, upper bound = {mu_hat[player][arm] + np.sqrt(self.g_2/2**epoch)}, (count_g = {count_g}) S[player] = {self.S[player]}, lower bound = {mu_hat[player][arm] - np.sqrt(self.g_2/2**epoch)}")

                    if count_b >= self.env.M - len(self.G[player]):
                        to_remove_from_S.append(arm)
                        assert arm not in self.B[player]
                        to_append_to_B.append(arm)
                        #logger.info(f"----- t = {self.t}, user {player}: arm {arm} is Bad (count_b = {count_b}) S[player] = {self.S[player]}, upper bound = {mu_hat[player][arm] + np.sqrt(self.g_2/2**epoch)}")

                    elif count_g >= self.env.K - self.env.M - len(self.B[player]):
                        if mu_hat[player][arm] > self.mu_lb + 3*np.sqrt(self.g_2/2**epoch):
                            to_remove_from_S.append(arm)
                            assert arm not in self.G[player]
                            to_append_to_G.append(arm)
                            #logger.info(f"----- t = {self.t}, user {player}: arm {arm} is Golden (count_g = {count_g}) S[player] = {self.S[player]}, lower bound = {mu_hat[player][arm] - np.sqrt(self.g_2/2**epoch)}")

                for arm_remove in to_remove_from_S:
                    self.S[player].remove(arm_remove)
                for arm_append in to_append_to_B:
                    self.B[player].append(arm_append)
                for arm_append in to_append_to_G:
                    self.G[player].append(arm_append)
            epoch += 1
