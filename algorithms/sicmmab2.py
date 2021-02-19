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

class Sicmmab2(Algorithm):
    def __init__(self, environment,
                mu_min=None):
        super().__init__(environment=environment)
        self.mu_min = mu_min
        if mu_min is None:
            self.mu_min = self.env.mu[-1]
        self.T_c = np.ceil(np.log(self.env.T)/self.mu_min)
        self.T_0 = np.ceil(2400*np.log(self.env.T)/self.mu_min)
        #self.T_0 = np.ceil(200
         #                  *np.log(self.env.T)/self.mu_min)

    def __str__(self):
        return f"SIC-MMAB2"

    def reset(self):
        super().reset()

    def B_s(self, s):
        return 3*np.sqrt(np.log(self.env.T)/(2*s))

    def run(self):
        #logger.info(f"Running {str(self)}, T_c = {self.T_c}, mu_min = {self.mu_min}")
        arms_t = np.zeros((self.env.M,),dtype=int)

        ##### INITIALIZATION PHASE #####
        self.M_p = np.ones((self.env.M,),dtype=int)
        self.t_estM = np.zeros((self.env.M,),dtype=int)


        ## Musical chair
        #logger.info(f"****** Starting initialization phase... Musical chairs for {self.env.K*self.T_c} steps")
        ext_rank = -1*np.ones((self.env.M,),dtype=int) #external rank
        while self.t < self.env.K*self.T_c:
            for player in range(self.env.M):
                if ext_rank[player] == -1:
                    if self.t > 0 and rewards_t[player] >= 1:
                        ext_rank[player] = arms_t[player]
                        logger.debug(f" t={self.t}, MC: player {player} has fixed on {arms_t[player]}")
                    else:
                        arms_t[player] = np.random.choice(self.env.K)
                        #logger.debug(f" t={self.t}, MC: player {player} not fixz")
                else:
                    arms_t[player] = ext_rank[player]
                    #logger.debug(f" t={self.t}, MC: player {player} not fixz")

            # Pull arms
            arms_t = arms_t.astype(int)
            rewards_t, regret_t, collisions_t = self.env.draw(arms_t)
            logger.debug(f" t={self.t}, MC: draw arms_t {arms_t}, rewards_t: {rewards_t}")
            self.update_stats(arms_t=arms_t, rewards=rewards_t, regret_t=regret_t, collisions_t=collisions_t)
            self.t += 1

        #logger.info(f"\n******** t = {self.t},  MC finished,  pulls: \n {self.pulls} ")
        #logger.info(f"\n******** t = {self.t},  Starting estimate_M.....")

        self.pi = ext_rank.copy()
        #logger.info(f"Players start at positions: {self.pi}")
        estimate_M_r = np.zeros((self.env.M,))
        ## Estimate_M_no_sensing
        while self.env.K*self.T_c <= self.t < self.env.K*self.T_c + 2*self.env.K*self.T_c:
            for player in range(self.env.M):
                if self.t_estM[player]%self.T_c == 0:
                    self.M_p[player] += (estimate_M_r[player] == 0 and self.t_estM[player] > 0)
                    estimate_M_r[player] = 0
                    logger.debug(f" t={self.t}, Estimate_M: t_est = {self.t_estM[player]}, updating M_p of user {player}: {self.M_p[player]}")
                if self.t_estM[player] >= 2*ext_rank[player]*self.T_c:
                    if self.t_estM[player] == 2*ext_rank[player]*self.T_c:
                        logger.debug(f" t={self.t}, Estimate_M: t_est = {self.t_estM[player]}, player {player} starts hopping because {self.t_estM[player]} = {2*ext_rank[player]*self.T_c}, extrank player: {ext_rank[player]}")
                    if (self.t_estM[player]%self.T_c == 0):
                        self.pi[player] = (self.pi[player]+1)%self.env.K
                arms_t[player] = self.pi[player]
                self.t_estM[player] += 1

            # Pull arms
            arms_t = arms_t.astype(int)
            rewards_t, regret_t, collisions_t = self.env.draw(arms_t)
            #logger.debug(f" t={self.t}, Estimate_M: draw arms_t {arms_t}, rewards_t: {rewards_t}")
            self.update_stats(arms_t=arms_t, rewards=rewards_t, regret_t=regret_t, collisions_t=collisions_t)
            estimate_M_r += rewards_t
            self.t += 1


        #logger.info(f"******** End of initialization phase: estimation M_p = {self.M_p}")
        ##### END INITIALIZATION PHASE #####



        ##### EXPLO COM EXPLOIT PHASE #####
        self.fixed = np.ones((self.env.M,),dtype=int)*-1
        self.occ_fix = np.ones((self.env.M,),dtype=int)*-1
        self.decl = [[] for player in range(self.env.M)] #decl[player]: list of declared players
        self.T_d = np.zeros((self.env.M,))
        self.T_expl = np.zeros((self.env.M,))

        self.active_arms = [np.arange(self.env.K) for m in range(self.env.M)]
        self.small_s = np.zeros((self.env.M, self.env.K)) # individual successes for each arm
        self.big_S = np.zeros((self.env.M, self.env.K)) # statistics with other players updates
        self.big_T = np.zeros((self.env.M, self.env.K),dtype=int) # Nb times drawn sans compter MC phase
        self.small_t = np.zeros((self.env.M, self.env.K),dtype=int) # Nb times drawn sans compter MC phase
        self.small_d = [[] for player in range(self.env.M)]
        self.opt = [[] for player in range(self.env.M)]

        self.p = np.ones((self.env.M,), dtype=int)

        self.communication_phase = [None for i in range(self.env.M)]
        self.exploration_phase = ["start" for i in range(self.env.M)]
        self.never_entered_receive = True
        self.update_phase = [False for i in range(self.env.M)]
        self.expl_mc_fix = np.array(-1*np.ones(self.env.M),dtype=int)

        self.rej = [[] for player in range(self.env.M)]
        self.acc = [[] for player in range(self.env.M)]


        #logger.info(f"\n******** Starts  EXPLO COM EXPLOIT PHASE ")

        while self.env.K*self.T_c + 2*self.env.K*self.T_c <= self.t < self.env.T:
            for player in range(self.env.M):
                ## EXPLOITATION ##
                if self.fixed[player] != -1:
                    arms_t[player] = self.fixed[player]

                ## EXPLORATION ##
                elif self.exploration_phase[player] is not None:
                    self.T_expl[player] = len(self.active_arms[player])*2**self.p[player]*self.T_0
                    if self.exploration_phase[player] == "start":
                        if len(self.decl[player]) > 0: # there has been declared arms
                            self.exploration_phase[player] = ["mc", len(self.active_arms[player])*self.T_c]
                            self.expl_mc_fix[player] = -1
                        else: # no declared arms take last phase stats
                            self.big_S[player] += self.small_s[player]
                            self.big_T[player] += self.small_t[player]
                            self.T_expl[player] -= self.T_d[player]
                            self.exploration_phase[player] = ["hopping", self.T_expl[player]]
                        #logger.info(f"******* t={self.t}, user {player} starting exploration {self.exploration_phase[player][0].upper()}, set decl = {self.decl[player]}, for {self.exploration_phase[player][-1]} steps")

                    if self.exploration_phase[player][0] == "hopping":
                        arms_t[player] = self.active_arms[player][self.pi[player]]
                        tmp_pi =self.pi[player]
                        self.pi[player] = (self.pi[player]+1)%len(self.active_arms[player])
                        #logger.debug(f" t={self.t}, user {player} had pi={tmp_pi} EXPLORATION: hopping to {self.pi[player]} which corresponds to {self.active_arms[player][self.pi[player]]}")
                    elif self.exploration_phase[player][0] == "mc":
                        if self.expl_mc_fix[player] == -1:
                            self.pi[player] = np.random.choice(len(self.active_arms[player]))
                            arms_t[player] = self.active_arms[player][self.pi[player]]
                        else:
                            #logger.info(self.expl_mc_fix[player])
                            arms_t[player] = self.active_arms[player][self.expl_mc_fix[player]]



                ## COMMUNICATION ##
                elif self.communication_phase[player] is not None:
                    if self.communication_phase[player] == "start_com_phase":
                        self.T_d[player] = len(self.active_arms)*self.T_0
                        self.decl[player] = []
                        self.rej[player] = []
                        self.acc[player] = []
                        #logger.info(f"******* t={self.t}, user {player} starting COMMUNICATION for round {self.p[player]}")
                        for arm_k in self.active_arms[player]:
                            ub_k = self.big_S[player][arm_k]/self.big_T[player][arm_k] + self.B_s(self.big_T[player][arm_k])
                            lb_k = self.big_S[player][arm_k]/self.big_T[player][arm_k] - self.B_s(self.big_T[player][arm_k])
                            logger.debug(f"arm_k lb = {lb_k}, ub = {ub_k}")
                            count_rej = np.sum((self.big_S[player,self.active_arms[player]]/self.big_T[player,self.active_arms[player]] \
                                               - self.B_s(self.big_T[player,self.active_arms[player]])) >= ub_k)
                            count_acc = np.sum(lb_k >= (self.big_S[player,self.active_arms[player]]/self.big_T[player,self.active_arms[player]] \
                                               + self.B_s(self.big_T[player,self.active_arms[player]])) )

                            logger.debug(f"count_acc = {count_acc}, count_rej = {count_rej}")

                            if count_rej >= self.M_p[player]:
                                self.rej[player].append(arm_k)
                            if count_acc >= len(self.active_arms[player]) - self.M_p[player]:
                                self.acc[player].append(arm_k)
                        #logger.info(f" t={self.t}, communication user {player} rej : {self.rej[player]}, acc: {self.acc[player]}")
                        self.communication_phase[player] = "choose_phase"

                    if self.communication_phase[player] == "choose_phase":
                        diff_rej_decl = np.setdiff1d(self.rej[player], self.decl[player])
                        diff_acc_decl = np.setdiff1d(self.acc[player], self.decl[player])
                        self.small_s[player] = np.zeros((self.env.K,))
                        self.small_t[player] = np.zeros((self.env.K,))

                        if len(diff_rej_decl) > 0:
                            self.communication_phase[player] = ["declare", diff_rej_decl[0], self.T_d[player]]
                            #logger.info(f" t={self.t}, communication user {player} enters mode: {self.communication_phase[player][0].upper()} for {self.communication_phase[player][-1]} steps. Declaring arm {diff_rej_decl[0]}")
                        elif len(diff_acc_decl) > 0:
                            self.communication_phase[player] = ["occupy", diff_acc_decl, self.T_d[player]]
                            self.occ_fix[player] = -1
                            #logger.info(f" t={self.t}, communication user {player} enters mode: {self.communication_phase[player][0].upper()} for {self.communication_phase[player][-1]} steps. Set_A = {diff_acc_decl}")
                        elif self.never_entered_receive or len(self.small_d[player]) > 0:
                            self.communication_phase[player] = ["receive", self.T_d[player]]
                            #logger.info(f" t={self.t}, communication user {player} enters mode: {self.communication_phase[player][0].upper()} for {self.communication_phase[player][-1]} steps.")

                    # COMMUNICATION: DECLARE PHASE
                    if self.communication_phase[player][0] == "declare":
                        declare_arm = self.communication_phase[player][1]
                        arms_t[player] = np.random.choice([declare_arm, self.active_arms[player][self.pi[player]]])
                        self.pi[player] = (self.pi[player] + 1)%len(self.active_arms[player])

                    # COMMUNICATION: OCCUPY PHASE
                    elif self.communication_phase[player][0] == "occupy":
                        set_A = self.communication_phase[player][1]
                        if self.occ_fix[player] == -1:
                            arms_t[player] = self.active_arms[player][self.pi[player]]
                        else:
                            arms_t[player] = self.occ_fix[player]
                        self.pi[player] = (self.pi[player] + 1)%len(self.active_arms[player])

                    # COMMUNICATION: RECEIVE PHASE
                    elif self.communication_phase[player][0] == "receive":
                        arms_t[player] = self.active_arms[player][self.pi[player]]
                        self.pi[player] = (self.pi[player] + 1)%len(self.active_arms[player])





            # Pull arms
            arms_t = arms_t.astype(int)
            rewards_t, regret_t, collisions_t = self.env.draw(arms_t)
            if self.t%70 == 0:
                logger.debug(f" t={self.t}, round p={self.p}, EXPLO COM EXPLORE: draw arms_t {arms_t}, rewards_t: {rewards_t}")
            self.update_stats(arms_t=arms_t, rewards=rewards_t, regret_t=regret_t, collisions_t=collisions_t)
            self.t += 1
            for player in range(self.env.M): # update and change phase selon reward
                if self.fixed[player] == -1:
                    # EXPLORE
                    if self.exploration_phase[player] is not None:
                        if self.exploration_phase[player][0] == "hopping":
                            self.big_S[player][arms_t[player]] += rewards_t[player]
                            self.big_T[player][arms_t[player]] += 1
                        elif self.exploration_phase[player][0] == "mc" and self.expl_mc_fix[player] == -1 and rewards_t[player] >= 1:
                            self.expl_mc_fix[player] = self.pi[player]
                            logger.debug(f" t={self.t}, exploration MC: player {player} fixed on {arms_t[player]}")
                        self.exploration_phase[player][1] -= 1
                        if self.exploration_phase[player][1] == 0:
                            if self.exploration_phase[player][0] == "hopping":
                                #logger.info(f"--------- t={self.t}, user {player} end of exploration phase, will start communication in next round")
                                self.exploration_phase[player] = None
                                self.communication_phase[player] = "start_com_phase"
                            elif self.exploration_phase[player][0] == "mc":
                                #logger.info(f"--------- t={self.t}, user {player} end of explo MC has fixed on {self.expl_mc_fix[player]}: starting hopping for {self.T_expl[player]} steps ")
                                self.exploration_phase[player] = ["hopping", self.T_expl[player]]

                    # COMMUNICATION
                    elif self.communication_phase[player][0] == "declare":
                        self.small_s[player][arms_t[player]] += rewards_t[player]
                        self.small_t[player][arms_t[player]] += 1
                        self.communication_phase[player][2] -= 1
                        if self.communication_phase[player][2] == 0:
                            self.small_d[player] = []
                            for i_arm in self.active_arms[player]:
                                if np.abs(self.big_S[player][i_arm]/self.big_T[player][i_arm] \
                                          - self.small_s[player][i_arm]/self.small_t[player][i_arm]) >= self.big_S[player][i_arm]/(4*self.big_T[player][i_arm]):
                                    self.small_d[player].append(i_arm)
                            self.small_d[player].append(declare_arm)
                            self.decl[player] = np.array(np.union1d(self.decl[player], self.small_d[player]),dtype=int)
                            self.communication_phase[player] = "choose_phase"
                            #logger.info(f" t={self.t}, communication: end of DECLARE phase of user {player} who declared {declare_arm}. Now set d: {self.small_d[player]}, Set decl: {self.decl[player]}")

                    elif self.communication_phase[player][0] == "occupy":
                        if self.occ_fix[player] == -1 and arms_t[player] in set_A and rewards_t[player] > 0:
                            self.occ_fix[player] = arms_t[player]
                            self.fixed[player] = arms_t[player]
                            logger.debug(f"user {player} was in OCCUPY phase, has fixed on {self.occ_fix[player]}")
                        self.small_s[player][arms_t[player]] += rewards_t[player]
                        self.small_t[player][arms_t[player]] += 1
                        self.communication_phase[player][2] -= 1
                        if self.communication_phase[player][2] == 0:
                            self.small_d[player] = []
                            for i_arm in self.active_arms[player]:
                                if np.abs(self.big_S[player][i_arm]/self.big_T[player][i_arm] \
                                          - self.small_s[player][i_arm]/self.small_t[player][i_arm]) >= self.big_S[player][i_arm]/(4*self.big_T[player][i_arm]):
                                    self.small_d[player].append(i_arm)
                            self.decl[player] =  np.array(np.union1d(self.decl[player], self.small_d[player]), dtype=int)
                            self.communication_phase[player] = "choose_phase"
                            #logger.info(f" t={self.t}, communication: end of OCCUPY phase of user {player}. Occupation: {self.occ_fix[player]}. Now set d for this phase: {self.small_d[player]}. Set decl: {self.decl[player]}")

                    elif self.communication_phase[player][0] == "receive":
                        self.small_s[player][arms_t[player]] += rewards_t[player]
                        self.small_t[player][arms_t[player]] += 1
                        self.communication_phase[player][1] -= 1
                        if self.communication_phase[player][1] == 0:
                            self.small_d[player] = []
                            for i_arm in self.active_arms[player]:
                                if np.abs(self.big_S[player][i_arm]/self.big_T[player][i_arm] \
                                          - self.small_s[player][i_arm]/self.small_t[player][i_arm]) >= self.big_S[player][i_arm]/(4*self.big_T[player][i_arm]):
                                    self.small_d[player].append(i_arm)
                                    logger.debug(f"Big S stats for arm  {i_arm}: {self.big_S[player][i_arm]/self.big_T[player][i_arm]} far from received in this phase: {self.small_s[player][i_arm]/self.small_t[player][i_arm]}, adding {i_arm} to d: now d is {self.small_d[player]}")
                                elif i_arm==2:
                                    logger.debug(f"arm {i_arm}, Big S stats {self.big_S[player][i_arm]/self.big_T[player][i_arm]} like reward received in this phase: {self.small_s[player][i_arm]/self.small_t[player][i_arm]}")
                            self.small_d[player] =  np.array(np.setdiff1d(self.small_d[player], self.decl[player]),dtype=int)
                            self.decl[player] = np.array(np.union1d(self.decl[player], self.small_d[player]),dtype=int) #self.decl[player] + list(self.small_d[player])
                            #logger.info(f" t={self.t}, communication: end of RECEIVE phase of user {player}. set d without previously declared: {self.small_d[player]}. Now decl set: {self.decl[player]}.")

                            if len(self.small_d[player]) > 0:
                                self.communication_phase[player] = "choose_phase"
                            else:# UPDATE PHASE
                                #logger.info(f" t={self.t}, round p ={self.p[player]}, user {player} UPDATE...")
                                for i_arm in self.decl[player]:
                                    if self.small_s[player][i_arm] == 0:
                                        self.opt[player].append(i_arm)
                                        #logger.info(f"...... now update: {i_arm} added to optimal arms, optimal arms: {self.opt[player]}")
                                self.active_arms[player] = np.array(np.setdiff1d(self.active_arms[player], self.decl[player]),dtype=int)
                                #logger.info(f"...... now active arms: {self.active_arms[player]}" )
                                self.M_p[player] = self.M_p[player] - len(self.opt[player])
                                #logger.info(f"...... now estimated M_p: {self.M_p[player]}" )
                                self.p[player] += 1
                                self.exploration_phase[player] = "start"
                                self.communication_phase[player] = None
