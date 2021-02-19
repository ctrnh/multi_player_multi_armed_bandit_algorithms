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

class Sicmmab(Algorithm):
    def __init__(self, environment,
                mu_min=None):
        super().__init__(environment=environment)
        self.mu_min = mu_min
        if mu_min is None:
            self.mu_min = self.env.mu[-1]
        #self.T_c = np.ceil(np.log(self.env.T)/self.mu_min)
        self.T_0 = np.ceil(self.env.K * np.log(self.env.T))
        self.collision_sensing = True
        self.sensing = False
        ##logger.info("T_0 is ", self.T_0)

    def __str__(self):
        return f"SIC-MMAB"


    def reset(self):
        super().reset()
        self.fixed = np.ones((self.env.M,),dtype=int)*-1
        self.M_hat = np.ones((self.env.M,),dtype=int)
        self.internal_ranks = np.zeros((self.env.M,), dtype=int)
        self.t_estM = np.zeros((self.env.M,),dtype=int)

        #self.active_arms[player] = np arr (K_p,) with idx of active arms


        self.communication_phase = [False for i in range(self.env.M)]
        self.exploration_phase = [True for i in range(self.env.M)]
        self.t_explo = np.zeros((self.env.M,),dtype=int)
        self.t_com_p = np.zeros((self.env.M),dtype=int)

    def define_Ep(self):
        # self.Ep[player] = [[i,l,k],..]
        self.Ep = [[] for player in range(self.env.M)]
        for player in range(self.env.M):
            for i in range(self.M_p[player]): # M_p
                for l in range(self.M_p[player]):
                    for k in range(len(self.active_arms[player])):
                        if i!=l:
                            self.Ep[player].append([i,l,k])


    def B_s(self, s):
        return 3*np.sqrt(np.log(self.env.T)/(2*s))

    def run(self):
        arms_t = np.zeros((self.env.M,),dtype=int)

        # Musical chair
        ext_rank = -1*np.ones((self.env.M,)) #external rank
        while self.t < self.T_0:
            for player in range(self.env.M):
                if ext_rank[player] == -1:
                    if self.t > 0 and np.sum(collisions_t[player,:]) == 0:
                        ext_rank[player] = arms_t[player]
                       # logger.debug(f"MC: player {player} fixed on external rank {arms_t[player]}")
                    else:
                        arms_t[player] = np.random.choice(self.env.K)
                else:
                    arms_t[player] = ext_rank[player]
            # Pull arms
            arms_t = arms_t.astype(int)
            #logger.debug(f"arms_t {arms_t}")
            rewards_t, regret_t, collisions_t = self.env.draw(arms_t)
            self.update_stats(arms_t=arms_t, rewards=rewards_t, regret_t=regret_t, collisions_t=collisions_t)
            self.t += 1

        #logger.info(f"********  MC finished, t = {self.t}, pulls: \n {self.pulls} ")


        self.pi = -1*np.ones((self.env.M,))
        # Estimate_M
        while self.T_0 <= self.t < self.T_0 + self.env.K*2:
            for player in range(self.env.M):
                if self.t_estM[player] == 0:
                    self.pi[player] = ext_rank[player] #k
                if self.t_estM[player] < 2*ext_rank[player]:
                    #logger.debug(f"since t_est {self.t_estM[player]} < 2*k {2*ext_rank[player]}, player {player} doesnt move ")
                    if self.t_estM[player] > 0 and np.sum(collisions_t[player,:]) >= 1:

                        self.M_hat[player] += 1
                        #logger.debug(f"passive collision at previous step, player {player} increase {self.M_hat}" )
                        self.internal_ranks[player] += 1
                    arms_t[player] = self.pi[player]
                else:
                    #logger.debug(f"since t_est {self.t_estM[player]} >= 2*k {2*ext_rank[player]}, player {player} goes next" )

                    if self.t_estM[player] > 2* ext_rank[player]:
                        if np.sum(collisions_t[player,:]) >= 1:
                            self.M_hat[player] += 1
                            #logger.debug(f"active collision at previous step, player {player} increase {self.M_hat}" )

                    self.pi[player] = (self.pi[player]+1)%self.env.K
                    arms_t[player] = self.pi[player]
                    #logger.debug(f"player {player}, pi: {self.pi[player]}")
                self.t_estM[player] += 1
            # Pull arms
            #logger.debug(f"arms_t = {arms_t}")
            arms_t = arms_t.astype(int)
            rewards_t, regret_t, collisions_t = self.env.draw(arms_t)
            self.update_stats(arms_t=arms_t, rewards=rewards_t, regret_t=regret_t, collisions_t=collisions_t)
            self.t += 1

        self.internal_ranks = self.internal_ranks.astype(int)
        #logger.info(f"******** Estimated ext_rank {ext_rank}, int rank {self.internal_ranks}, M_hat {self.M_hat}")


        self.p = np.ones((self.env.M,), dtype=int)
        self.M_p = self.M_hat.copy().astype(int) # (M,) self.M_p[player]: nb of active players estimated by player
        self.active_arms = [np.arange(self.env.K) for m in range(self.env.M)]
        self.s = np.zeros((self.env.M, self.env.K),dtype=int) # individual successes for each arm
        self.S_tilde = np.zeros((self.env.M, self.env.K),dtype=int) # statistics with other players updates
        self.T_ = np.zeros((self.env.M, self.env.K),dtype=int) # Nb times drawn sans compter MC phase
        self.com_state = [None for player in range(self.env.M)]

        self.pi_explore = np.zeros((self.env.M),dtype=int) # idx in terms of self.active_arms

        #logger.info(f"******** Starts explocomexploit")

        # Exploration/com/exploit
        while self.T_0 + self.env.K*2 <= self.t < self.env.T:
            for player in range(self.env.M):
                #Exploit
                if self.fixed[player] != -1:
                    arms_t[player] = self.fixed[player]

                #Explore/Communicates
                else:
                    # exploration phase
                    if self.exploration_phase[player]:
                        if self.t_explo[player] == 0:
                            self.pi_explore[player] = int(self.internal_ranks[player]) # j-th
                        else:
                            self.s[player, self.active_arms[player][self.pi_explore[player]]] += rewards_t[player] # update individual successes
                        self.pi_explore[player] = int((self.pi_explore[player]+1)%len(self.active_arms[player])) # Sequential hopping
                        self.pi_explore = self.pi_explore.astype(int)
                       # logger.debug(f"t_explo={self.t_explo[player]}: player {player} ({self.internal_ranks[player]}) in exploration phase plays {self.pi_explore[player]}-th active arm")
                        arms_t[player] = self.active_arms[player][self.pi_explore[player]]
                        self.t_explo[player] += 1

                        if self.t_explo[player] == len(self.active_arms[player])*2**self.p[player]:
                            self.communication_phase[player] = True
                            self.exploration_phase[player] = False
                            self.com_state = [None for player in range(self.env.M)]
                            self.triplet_idx = [0 for player in range(self.env.M)]
                            self.t_explo[player] = 0
                            #logger.info(f"--------- end of exploration phase, will start communication in next round")
                            #logger.info(f"--------- individual success received by p {player}({self.internal_ranks[player]}) are {self.s[player]}")

                    # communication phase
                    elif self.communication_phase[player]:
                        if self.t_com_p[player] == 0:
                            self.s[player, self.active_arms[player][self.pi_explore[player]]] += rewards_t[player] # update last success of explo phase
                            #logger.info(f"player {player} ({self.internal_ranks[player]}) enters communication, t={self.t}. Defining Ep")
                            self.define_Ep() # 1st time communicate at self.p
                            self.S_tilde = self.s.copy()
                        #logger.debug(f"self.Ep {self.Ep}")
                        #logger.debug(f"triplet_idx {self.triplet_idx}")

                        if self.M_p[player] != 1:
                            i, l, k = self.Ep[player][self.triplet_idx[player]] # current triplet


                            if self.com_state[player] is None:
                                if i == self.internal_ranks[player]:
                                    self.com_state[player] = ["send",0]
                                elif l == self.internal_ranks[player]:
                                    self.com_state[player] = ["receive",0] #(_,n)
                                else:
                                    self.com_state[player] = ["wait",0]
                                #logger.debug(f"Current triplet is {(i, l, k)}, player int rank{self.internal_ranks[player]} has state {self.com_state[player]}")

                            # Communicates: send
                            if self.com_state[player][0] == "send":
                                s_k = int(self.s[player,self.active_arms[player][k]]) # stats of the player for k-th active arm (maxi = 2**p)
                                bin_s_k = (bin((s_k)).split('b')[-1]).zfill(self.p[player]+1) # convert binary
                                #logger.debug(f"{i} wants to send {bin_s_k} corresponding to {s_k}")
                                if int(bin_s_k[-1-self.com_state[player][1]]): #send 1 if n-th binary term = 1
                                    #logger.debug(f"p int rank {i} sends 1 to int rank{l},{self.active_arms[player][l]} about {k}th active arm")
                                    arms_t[player] = self.active_arms[player][l]
                                else:
                                    arms_t[player] = self.active_arms[player][self.internal_ranks[player]]

                            # Communicates: receive
                            elif self.com_state[player][0] == "receive":
                                arms_t[player] = self.active_arms[player][self.internal_ranks[player]]

                            # Communicates: wait
                            elif self.com_state[player][0] == "wait":
                                arms_t[player] = self.active_arms[player][self.internal_ranks[player]]


                        self.t_com_p[player] += 1


            # Pull arms
            arms_t = arms_t.astype(int)
            #if self.t%100 == 0:
                #logger.info(f"t = {self.t}, arms_t {arms_t}, fixed : \n {self.fixed}")
            rewards_t, regret_t, collisions_t = self.env.draw(arms_t)
            self.update_stats(arms_t=arms_t, rewards=rewards_t, regret_t=regret_t, collisions_t=collisions_t)
            self.t += 1

            for player in range(self.env.M):
                if self.fixed[player] ==-1 and self.communication_phase[player] and self.t_com_p[player] > 0:
                    if self.M_p[player] != 1:
                        i, l, k = self.Ep[player][self.triplet_idx[player]] # current triplet
                        #logger.debug(f"triplet {(i,l,k)}")

                        if self.com_state[player][0] == "receive":
                            if np.sum(collisions_t[player,:]) > 0:
                                #logger.debug(f"p {player} of in rank {self.internal_ranks[player]} receives 1 about {k}-th active arm")
                                self.S_tilde[player][self.active_arms[player][k]] += 2**self.com_state[player][1]

                        #logger.debug(f"1com_state {self.com_state[player]}, p = {self.p[player]}")

                        if self.com_state[player][1] == self.p[player]: # finish communication of the triplet
                            self.com_state[player] = None
                            self.triplet_idx[player] += 1 # go to next triplet
                        else:
                            self.com_state[player][1] += 1 # increment n


                    # check if visited all triplets (end communication)
                    if self.triplet_idx[player] == len(self.Ep[player]): # no next triplet
                        #logger.info(f".............finished all triplets, of  p: {self.p}th explo/com phase")
                        self.communication_phase[player] = False
                        self.exploration_phase[player] = True # Go back to exploration phase
                        self.com_state[player] = None
                        self.triplet_idx[player] = 0
                        self.t_com_p[player] = 0

                        # Update T[k]
                        self.T_[player] += self.M_p[player]*2**self.p[player]
                        logger.debug(f"self.T_ {self.T_}")
                        logger.debug(f"self.S_tilde {self.S_tilde}")
                        #Update Stats of player

                        rej = []
                        acc = []
                        for arm_k in self.active_arms[player]:
                            ub_k = self.S_tilde[player][arm_k]/self.T_[player][arm_k] + self.B_s(self.T_[player][arm_k])
                            lb_k = self.S_tilde[player][arm_k]/self.T_[player][arm_k] - self.B_s(self.T_[player][arm_k])
                            logger.debug(f"arm_k lb = {lb_k}, ub = {ub_k}")
                            count_rej = np.sum((self.S_tilde[player,self.active_arms[player]]/self.T_[player,self.active_arms[player]] \
                                               - self.B_s(self.T_[player,self.active_arms[player]])) >= ub_k)
                            count_acc = np.sum(lb_k >= (self.S_tilde[player,self.active_arms[player]]/self.T_[player,self.active_arms[player]] \
                                               + self.B_s(self.T_[player,self.active_arms[player]])) )

                            logger.debug(f"count_acc = {count_acc}, count_rej = {count_rej}")

                            if count_rej >= self.M_p[player]:
                                rej.append(arm_k)
                            if count_acc >= len(self.active_arms[player]) - self.M_p[player]:
                                acc.append(arm_k)
                            logger.debug(f"rej {rej}, acc {acc}")
                        if self.M_p[player] - self.internal_ranks[player] <= len(acc):
                            self.fixed[player] = acc[self.M_p[player] -  self.internal_ranks[player]-1]
                            logger.debug(f"88888888888888888888888888 Player {player} IS FiXED ")
                        else:
                            self.M_p[player] -= len(acc)

                            self.active_arms[player] = np.array(np.setdiff1d(self.active_arms[player],
                                                                   np.unique(acc+rej)),dtype=int)
                            #logger.debug(f"new set of active arms of p {player}: {self.active_arms[player]}")
                            #logger.debug(f"estimated active players {self.M_p[player]}")

                        self.p[player] += 1
                        #logger.info(f"New active arms: {self.active_arms}")
                        #logger.info(f"Estimated active players {self.M_p}")
                        #logger.info(f"Fixed: {self.fixed}")
