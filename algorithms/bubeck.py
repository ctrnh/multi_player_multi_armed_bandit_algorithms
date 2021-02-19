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

class Bubeck(Algorithm):
    def __init__(self, environment):
        super().__init__(environment=environment)
        assert self.env.M == 2 and self.env.K == 3
        self.t0 = 40*np.sqrt(self.env.T*np.log(self.env.T))
        self.k0 = np.floor(np.log2(self.t0))

        self.c = 0
        self.klucb = np.vectorize(klucb_ber)
        #logger.info(f't_0={self.t0}, should be approx = 2**k0 = {2**self.k0}, k_0 = {self.k0}')
        self.t0 = 2**self.k0
    def __str__(self):
        return f"Bubeck-Budzinski"


    def compute_ucb_idx(self):
        ucb_idx = np.zeros((self.env.M,
                            self.env.K))

        d = (np.log(self.t) + self.c*np.log(np.log(self.t))) / (self.pulls+1e-7)
        ucb_idx = self.klucb(self.mu_hat, d)
        ucb_idx[self.pulls < 1] = float('+inf')
        return ucb_idx


    def run(self):
        fixed = -1*np.ones(self.env.M)
        arms_t = -1*np.ones(self.env.M)
        #logger.info(f"\n******************* Running with {self.env.mu}, T = {self.env.T}\n" )
        k = self.k0
        while self.t <= 2**k:
            arms_t[0] = 2 # Alice stays on 3
            arms_t[1] = int(self.t%2 == 0) # Bob alternates between 1 and 2
            arms_t = arms_t.astype(int)
            rewards_t, regret_t, collisions_t = self.env.draw(arms_t)
            self.update_stats(arms_t=arms_t, rewards=rewards_t, regret_t=regret_t, collisions_t=collisions_t)
            self.env.update(t=self.t)
            self.t += 1

        set_B = {2} # 3 is always in B
        for i in range(2):
            if self.mu_hat[1,i] > np.sqrt(np.log(self.env.T)/self.env.T):
                set_B.add(i)

        #logger.info(f"......t = {self.t}, init phase finished. pulls: \n {self.pulls}, mu_hat = \n{self.mu_hat}. \n Current set_B: {set_B}, sqrtlogT/T={np.sqrt(np.log(self.env.T)/self.env.T)}")

        while self.t < self.env.T:
            arms_t = fixed.copy()

            if self.t == 2**k+1:
                k += 1
                #logger.info(f"\n \n***** t = {self.t}, Starting new phase k = {k}. \npulls = {self.pulls}\n, mu_hat = \n {self.mu_hat}, set_B = {set_B}\n")
                # Alice
                if fixed[0] == -1:
                    # (i)
                    if np.abs(self.mu_hat[0,0] - self.mu_hat[0,1]) >= 10*np.sqrt(np.log(self.env.T)/(self.t-self.t0)):
                        if k-self.k0 >=3:
                            fixed[0] = np.argmax(self.mu_hat[0,:2])
                            #logger.info(f"\n ........t = {self.t}, k = {k-1}, Alice is fixed on {fixed[0]} (mu_hat = {self.mu_hat[0,:]})\n , pulls = \n {self.pulls[0,:]}")

                # bob
                if fixed[1] == -1:
                    ## iii
                    if len(set_B) == 3: # B = {0,1,2}
                        if self.mu_hat[1,2] <= min(self.mu_hat[1,:2]) - 100*np.sqrt(np.log(self.env.T)/self.t):
                            set_B.remove(2)
                            #logger.info(f"\n .........t = {self.t}, k = {k-1}, Arm 2 is removed from set_B. Indeed, mu_hat = {self.mu_hat[1,:]}")
                        #else:
                            #logger.info(f"\n .........t = {self.t}, k = {k-1}, set_B is still length 3, arm 3 not removed: mu_hat = {self.mu_hat[1,:]}, conf = {100*np.sqrt(np.log(self.env.T)/self.t)} ")

                    ## iv
                    fixed[1] = 2
                    for j in set_B.intersection({0,1}):
                        if self.mu_hat[1,2] < self.mu_hat[1,j] + 100*np.sqrt(np.log(self.env.T)/self.t):
                            fixed[1] = -1

                    #if fixed[1] == 2:
                        #logger.info(f"\n ....t = {self.t}, k = {k-1}, Bob is fixed on 2")
                    rewards_phase_bob = np.zeros(3)

            if fixed[0] == -1: # alice
                arms_t[0] = int(k%2==0)
            else:
                arms_t[0] = fixed[0]


            if fixed[1] == -1: # bob
                i_k = int(k%2==0)#int(self.t%2==0) #int(k%2==0) # = 1 if k even
                valid_actions_list = sorted(list(set_B.difference({i_k})))
                if len(valid_actions_list) == 1:
                    arms_t[1] = valid_actions_list[0]
                else:
                    arms_t[1] = valid_actions_list[self.t%2==0]

                if self.t == np.ceil(2**(k-1) + 40*np.sqrt(self.env.T*np.log(self.env.T))):
                    #logger.info(f"...........t = {self.t}, k = {k}, rewards_phase_bob = {rewards_phase_bob}")
                    for action in valid_actions_list:
                        if action != 2 and rewards_phase_bob[action] == 0:

                            #logger.info(f"..........Bob has received only 0 from arm {action} so from now on, bob plays single MAB")
                            fixed[1] = -2
                            self.pulls[1,:] = 0
                            self.successes[1,:] = 0
                            self.mu_hat[1,:] = 0
            elif fixed[1] == -2: # bob plays single MAB (KLUCB)
                ucb_idx_bob = self.compute_ucb_idx()[1,:]
                arms_t[1] = best_arm_rnd(ucb_idx_bob)
            else:
                arms_t[1] = fixed[1]

            arms_t = arms_t.astype(int)
            rewards_t, regret_t, collisions_t = self.env.draw(arms_t)
            rewards_phase_bob[arms_t[1]] += rewards_t[1]
            self.update_stats(arms_t=arms_t, rewards=rewards_t, regret_t=regret_t, collisions_t=collisions_t)
            self.env.update(t=self.t)
            self.t += 1
