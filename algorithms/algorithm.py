import numpy as np
import matplotlib.pyplot as plt

import collections
import os

import logging
import pickle

import time
logger = logging.getLogger(__name__)
from tqdm import tqdm


class Algorithm:
    def __init__(self, environment, save_optional=False):
        self.env = environment
        self.sensing = False
        self.collision_sensing = False

        self.save_optional = save_optional
        self.reset()

    def reset(self):
        self.t = 0
        self.regret = np.zeros(self.env.T)
        self.pulls = np.zeros((self.env.M,
                               self.env.K),  dtype=np.int32)
        self.successes = np.zeros((self.env.M,
                                   self.env.K))
        self.mu_hat = np.zeros((self.env.M,
                               self.env.K))


        self.collisions = np.zeros((self.env.M, self.env.K)) # self.collisions[i,j] = nb of collisions of player i on arm j

        self.collision_hist = np.zeros((self.env.K, self.env.T))

        self.delta_hist = []
        self.success_hist = []
        self.pulls_hist = []

        if self.env.dynamic:
            self.sum_mu_opt = 0
            self.system_reward_tot = 0


        if self.save_optional:
            self.arm_history = np.zeros((self.env.M, self.env.T))
        if self.env.K == 2 and self.save_optional:
            self.delta_hist = np.zeros((self.env.M, self.env.T))
            self.success_hist = np.zeros((self.env.M,self.env.K, self.env.T))
            self.pulls_hist = np.zeros((self.env.M, self.env.K, self.env.T))

    def arms_t_policy(self):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError
        self.reset()
        while self.t < self.env.T:
            arms_t = self.arms_t_policy()
            rewards_t, regret_t ,collisions_t = self.env.draw(arms_t,)

            self.update_stats(arms_t=arms_t, rewards=rewards_t, regret_t=regret_t, collisions_t=collisions_t)
            self.env.update(t=self.t)
            self.t += 1


    def compute_mu_hat(self):
        mu_hat = np.zeros((self.env.M, self.env.K))
        if self.collision_sensing and not self.sensing:
            mu_hat = self.successes / (self.pulls - self.collisions)
            mu_hat[self.successes==0] = 0# ie mu_hat[self.pulls == self.collisions] = 0
        else:
            mu_hat = self.successes / (1e-7+self.pulls)
        mu_hat[self.pulls == 0] = 0 # if arm has never been pulled, mu_hat =0
        return mu_hat

    def update_stats(self, arms_t, rewards, regret_t, collisions_t, system_reward=0):
        """
        arms_t: vector of size env.M (number of players currently in the game)
        """
        if self.t < self.env.T:
            self.regret[self.t] = regret_t
            if self.env.dynamic:
                list_active_players = list(self.env.active_players)
                self.pulls[list_active_players, arms_t[list_active_players]] += 1
                self.successes[list_active_players, arms_t[list_active_players]] += rewards[list_active_players]
                self.mu_hat = self.compute_mu_hat()

                self.collisions += collisions_t
                self.collision_hist[:,self.t] += np.max(collisions_t, axis=0)

                self.sum_mu_opt += self.env.mu_opt
                self.system_reward_tot += system_reward

            else:
                self.pulls[np.arange(self.env.M), arms_t] += 1
                self.successes[np.arange(self.env.M), arms_t] += rewards
                self.mu_hat = self.compute_mu_hat()

                self.collisions += collisions_t
                self.collision_hist[:,self.t] += np.max(collisions_t, axis=0)

            if self.env.K == 2 and self.save_optional:
                self.arm_history[:self.env.M,self.t] = arms_t
                self.pulls_hist[:,:,self.t] = self.pulls
                self.delta_hist[:, self.t] = self.mu_hat[:,0] - self.mu_hat[:,1]
                self.success_hist[:,:, self.t] = self.successes



    def reset_player(self, leaving_players):
        """
        (dynamic setting), when a player leaves, its statistics are resetted
        """
        for idx_player in leaving_players:
            self.mu_hat[idx_player,:] = 0
            self.pulls[idx_player,:] = 0
            self.successes[idx_player,:] = 0




    def plot_arm_history(self,
                         min_T=0,
                         max_T=None,
                        ):
        if max_T is None:
            max_T = self.env.T
        #fig = plt.figure(figsize=(12,2))
        #fig.add_subplot(111)
        for player in range(self.env.M):
            player_arm_history = list(self.arm_history[player,:])
            plt.step(np.arange(min_T,max_T),
                     player_arm_history[min_T:max_T],
                     where="post",
                     label=f"user {player}")

        plt.ylabel("arms")
        plt.xlabel("t")
        plt.legend(loc="lower right")


    def plot_delta_history(self,
                         min_T=0,
                         max_T=None,
                         y_min=-1,
                         y_max=1
                        ):
        if max_T is None:
            max_T = self.env.T

        name = f"delta"# T_{min_T}_{max_T}"


        for player in range(self.env.M):
            player_hist = self.delta_hist[player,:]
            plt.step(np.arange(min_T,max_T),
                     player_hist[min_T:max_T],
                     where="post",
                     label=f"user {player} ")
        plt.plot(np.arange(min_T,max_T),
                 np.zeros((max_T-min_T,)))
        plt.ylabel(name)
        plt.xlabel("t")
        plt.legend(loc="lower right")
        plt.title(name)
        plt.grid()
        plt.ylim(y_min,y_max)


    def plot_mu_hat(self,min_T,max_T):
        if max_T is None:
            max_T = self.env.T
        mu_hat_hist = self.success_hist/self.pulls_hist
        mu_hat_hist[self.pulls_hist == 0] = 0
        for arm in range(self.env.K):
            plt.plot(np.arange(min_T,max_T),
                     self.env.mu[arm] * np.ones((max_T-min_T,)),"--"
                    )
            for player in range(self.env.M):
                plt.step(np.arange(min_T,max_T),mu_hat_hist[player,arm,min_T:max_T],label=f"user {player}", where="post",)
        plt.plot(np.arange(min_T,max_T),
                 np.zeros((max_T-min_T,)),
                )
        plt.ylabel("mu_hat")
        plt.xlabel("t")
        plt.legend(loc="lower right")
        plt.grid()
