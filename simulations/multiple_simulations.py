import numpy as np
import matplotlib.pyplot as plt

import collections
import os

import logging
import pickle
import json
import time
logger = logging.getLogger(__name__)



from tqdm import tqdm, tqdm_notebook
import importlib

from environment import *
from simulations import *
from algorithms import *




def compute_memory(T, n_exps, pas=1):
    mb = round(T*n_exps*(35*1e-6),1)
    #logger.info(f'{mb} MB = {mb*1e-3} GB')
    mb /= pas
    #logger.info(f'{mb} MB = {mb*1e-3} GB')
    return mb*1e-3

class MultipleSimulations:
    """
    Helper class to run multiple algorihms in given environments
    """
    def __init__(self,
                 M,
                 dict_K_T,
                dict_given_list={},
                dynamic_params=None):
        """
        - dict_K_T: (dict)
            dict_K_T[K] = T (horizon for runs with K arms)
        - dict_given_list: (dict)
            dict_given_list[K] = [[mu_1, ..., mu_K], ....] (list of mus environments to test)
        """
        self.M = M
        self.dict_K_T = dict_K_T
        self.dict_given_list = dict_given_list
        self.dynamic_params = dynamic_params


    def output_list_mu(self, K,
                      ):
        list_mu = []
        if K in self.dict_given_list:
            list_mu = list_mu + self.dict_given_list[K]
        return list_mu

    def compute_list_env(self, print_env=False):
        self.list_env = []
        memory_one_exp_each = 0
        for K in self.dict_K_T:
            list_mu = self.output_list_mu(K=K)
            config = {}
            config['M'] = self.M
            config['horizon'] = self.dict_K_T[K]
            config['dynamic'] = False

            for mu in list_mu:
                config['mu'] = mu
                if not self.dynamic_params:
                    env = Environment(config=config,
                                     deterministic=False)
                else:
                    env = DynamicEnvironment(config=config, deterministic=False,
                                            players_can_leave=self.dynamic_params["can_leave"],
                                            t_entries=self.dynamic_params["t_entries"],
                                            t_leaving=self.dynamic_params["t_entries"],
                                            lambda_poisson=self.dynamic_params["lambda_poisson"],
                                            mu_exponential=self.dynamic_params["mu_exponential"])
                if print_env:
                    logger.info(env)
                self.list_env.append(env)
                memory_one_exp_each += compute_memory(T=env.T, n_exps=1,pas=50)
        logger.info("Nb env:", len(self.list_env))
        logger.info("One exp: ", memory_one_exp_each, 'GB')


    def run_save_fig(self, n_exps, list_algo_to_plot, skip_existing_sim=True):
        self.compute_list_env()
        for env in self.list_env:

            list_algo = [# No collision sensing information
                        Lugosi2(environment=env),
                        Sicmmab2(environment=env),
                        Selfishucb(environment=env,
                                    randomized=True
                                   ),
                        Lugosi1(environment=env),
                        Ecsic(environment=env),
                        Selfishucb(environment=env,
                                    randomized=False
                                           ),

                        # Collision sensing information
                        Sicmmab(environment=env),
                        MCTopM(environment=env)
                        ]
            if env.dynamic:
                list_algo.append(Dynmmab(environment=env))
            if env.M == 2 and env.K == 3:
                list_algo.append(Bubeck(environment=env),)



            new_list_algo = []
            all_algo_names_to_plot = []
            for algo in list_algo:
                if str(algo) in list_algo_to_plot:
                    new_list_algo.append(algo)
                    all_algo_names_to_plot.append(str(algo))
            assert len(new_list_algo) == len(list_algo_to_plot), "at least one algorithm is mispelled"

            logger.info(f"\n Running for M ={env.M}, K = {env.K} mu={env.mu}, algos:{new_list_algo}")
            sim = Simulation(environment=env, list_algo=new_list_algo, skip_existing_sim=skip_existing_sim)

            sim.run_save_hist(nb_exps=n_exps)
            sim.save_individual_figures_all(percentile=10)
