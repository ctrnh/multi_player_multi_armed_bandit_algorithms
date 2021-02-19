import numpy as np
import matplotlib.pyplot as plt

import collections
import os

import logging
import pickle
from tqdm import tqdm

logger = logging.getLogger(__name__)

class Environment:
    def __init__(self, config, deterministic=False):
        self.T = config['horizon']
        self.M = config['M']

        self.mu = config['mu']
        self.mu = np.array(sorted(self.mu, reverse=True))
        #self.mu_opt_M = np.sort(self.mu)[-self.M:]
        self.mu_opt = np.sum(self.mu[:self.M])
        self.K = len(self.mu)
        logger.info(f"Created environment with M = {self.M}, mu = {self.mu}, mu_opt = {self.mu_opt}, T = {self.T}, K = {self.K}")
        self.dynamic = False
        self.deterministic = deterministic

    def __str__(self):
        return f"M{self.M}-K{self.K}-mu{str(self.mu)}"

    def update(self, t):
        pass

    def draw(self, arms, sensing=False):
        counts = collections.Counter(arms)
        rewards = np.zeros((self.M,))
        regret_t = self.mu_opt
        collisions_t = np.zeros((self.M, self.K))
        for player in range(self.M):
            if counts[arms[player]] >= 2: # There was collision
                rewards[player] = 0
                collisions_t[player, arms[player]] = 1
            if counts[arms[player]] < 2 or sensing:
                if self.deterministic:
                    rewards[player] = self.mu[arms[player]]
                else:
                    rewards[player] = np.random.binomial(n=1, p=self.mu[arms[player]])
            if counts[arms[player]] < 2:
                regret_t -= self.mu[arms[player]]


        return rewards, regret_t, collisions_t


class DynamicEnvironment(Environment):
    def __init__(self,config, players_can_leave=False, deterministic=False,
                t_entries=None,
                t_leaving=None,
                lambda_poisson=1/10000,
                mu_exponential=10000):
        super().__init__(config=config, deterministic=deterministic)
        assert self.M == self.K
        self.dynamic = True
        self.players_can_leave = players_can_leave


        self.lambda_poisson = lambda_poisson
        self.mu_exponential = mu_exponential# how many steps do they stay? ~10 seconds  = 10k steps


        if t_entries:
            self.t_entries = t_entries
        else:
            self.sample_t_entries()
        self.t_entries.sort()
        self.t_entries[0] = 0
        print("Nb entrées:", len(self.t_entries))
        assert self.t_entries[1] != 0

        if players_can_leave:
            if t_leaving:
                self.t_leaving = t_leaving
            else:
                self.sample_t_leaving()
            if not (self.t_leaving > self.t_entries).all():
                self.t_leaving[np.argwhere(self.t_leaving == self.t_entries).flatten()] += 1

            assert (self.t_leaving > self.t_entries).all(), f"{self.t_entries[np.argwhere(self.t_leaving <= self.t_entries).flatten()]}, sorties:{self.t_leaving[np.argwhere(self.t_leaving <= self.t_entries).flatten()]}"
            assert len(np.unique(self.t_leaving) == len(self.t_leaving))
            self.t_leaving_dict = dict(zip(self.t_leaving, [None for i in range(len(self.t_leaving))]))
            self.t_leaving_dict[self.t_leaving[0]] = 0
        else:
            self.t_leaving = []#self.T*np.ones(self.M)
            self.t_leaving_dict = {}
            self.individual_horizons = np.zeros((self.M,))
            for player in range(self.M):
                self.individual_horizons[player] = self.T - self.t_entries[player] # il n'y a que les M premiers qui peuvent entrer dans le game

        self.active_players = {0}
        self.inactive_players = {i for i in range(1,self.M)}

        self.mu_opt = np.sum(self.mu[:len(self.active_players)])
        print("mu_optinit:",self.mu_opt)
        self.ith_entry = 1


    def sample_t_entries(self):
        n_entries = max(self.M, np.random.poisson(self.lambda_poisson*self.T))

        self.t_entries = np.random.randint(self.T, size=n_entries)


    def sample_t_leaving(self):
        staying_time = np.random.exponential(self.mu_exponential, size=len(self.t_entries))
        self.t_leaving = np.minimum(self.t_entries + (staying_time).astype(int), self.T)


    def update(self, t):
        '''
        Leaving players have played the round t
        '''
        env_has_changed = False

        leaving_players = []


        while t in self.t_leaving_dict:
            idx_player = self.t_leaving_dict[t]
            leaving_players.append(idx_player)
            self.active_players.remove(idx_player)
            self.inactive_players.add(idx_player)
            #print(f"........t = {t}: player of idx {idx_player} leaves the game")
            del self.t_leaving_dict[t]
            env_has_changed = True

        while self.ith_entry < len(self.t_entries) and t == self.t_entries[self.ith_entry]:
            if len(self.active_players) == self.M:
               # print(f"__t = {t}: saturated system, player at entry {t} cannot enter")
                if not self.players_can_leave:
                    return leaving_players
                if self.t_leaving[self.ith_entry] in self.t_leaving_dict:
                    del self.t_leaving_dict[self.t_leaving[self.ith_entry]]
                self.ith_entry += 1

            for idx_player in range(self.M):
                if idx_player not in self.active_players:
                    self.active_players.add(idx_player)
                    self.inactive_players.remove(idx_player)
                    if self.players_can_leave:
                        self.t_leaving_dict[self.t_leaving[self.ith_entry]] = idx_player
                    #print(f"........t = {t}: player {self.ith_entry} enters the game at player slot {idx_player}")
                    self.ith_entry += 1
                    break
            env_has_changed = True


        if env_has_changed:
            self.mu_opt = np.sum(self.mu[:len(self.active_players)])
            #print(f"........t = {t}: active players: {self.active_players} ({len(self.active_players)}). new mu_opt = {self.mu_opt}")

        return leaving_players





    def draw(self, arms):
        arms[list(self.inactive_players)] = -1
        counts = collections.Counter(arms)
        rewards = np.zeros((self.M,))
        regret_t = self.mu_opt
        collisions_t = np.zeros((self.M, self.K))
        for player in self.active_players:
            if counts[arms[player]] >= 2: # There was collision
                rewards[player] = 0
                collisions_t[player, arms[player]] = 1
            else:
                if self.deterministic:
                    rewards[player] = self.mu[arms[player]]
                else:
                    rewards[player] = np.random.binomial(n=1, p=self.mu[arms[player]])
                regret_t -= self.mu[arms[player]]#rewards[player]

        system_reward = self.mu_opt - regret_t
        #print("draw, ", arms, collisions_t)
        return rewards, regret_t, collisions_t, system_reward
