import numpy as np
import matplotlib.pyplot as plt

import collections
import os

import logging
import pickle
import json
import time



from tqdm import tqdm
import importlib

from environment import *
from simulations import *
from algorithms import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)




def print_results(algo,T):
    logger.info(f"\n  {str(algo)}")
    reward_opt_per_time = algo.sum_mu_opt/T
    sys_reward_per_time = algo.system_reward_tot/T
    print("sum mu_opt:", algo.sum_mu_opt, "per unit time:", reward_opt_per_time)
    print("system reward:", algo.system_reward_tot, "per unit time:", sys_reward_per_time)
    ratio =  sys_reward_per_time/reward_opt_per_time
    print("ratio:",ratio)
    return reward_opt_per_time, sys_reward_per_time, ratio


def compute_results_dict(env_config,
                        list_every_t,
                        list_mu_exponential,
                        algo,
                        n_exps,
                        save_folder):

    ## Creating dict to store results for each parameter
    results_dict = {}
    for every_t in list_every_t:
        results_dict[every_t] = {}
        for mu_exponential in list_mu_exponential:
            results_dict[every_t][mu_exponential] = []
    for every_t in list_every_t:
        for mu_exponential in list_mu_exponential:
            for n_exp in range(n_exps):
                print(f"*********** new player every {every_t}, staying time: {mu_exponential} ")
                env = DynamicEnvironment(config=env_config, players_can_leave=True, deterministic=False,
                t_entries=None,#t_entries_common,
                t_leaving=None,#t_leaving_common,
                lambda_poisson=1/every_t,
                mu_exponential=mu_exponential
                )

                if algo == "Rnd-SelfishKLUCB":
                    dynsucb = Selfishucb(environment=env,
                    randomized=True
                    )

                    dynsucb.run()
                    reward_opt_per_time, sys_reward_per_time, ratio = print_results(algo=dynsucb,T=env.T)
                    results_dict[every_t][mu_exponential].append(round(ratio,2))
                elif algo == "DYN-MC":
                    dynmc = Dynmc(environment=env,)
                    dynmc.run()
                    reward_opt_per_time, sys_reward_per_time, ratio = print_results(algo=dynmc,T=env.T)
                    results_dict[every_t][mu_exponential].append(round(ratio,2))

    filename = f"{algo}_T-{T}_K-{K}_multiple_lambda_mu_n_exps{n_exps}"
    json_path = os.path.join(save_folder, filename + ".json")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print(f"Create {save_folder} folder")
    with open(json_path,'w') as f:
        print("save",json_path)
        json.dump(results_dict , f)


    mean_std_dict = {}
    for every_t in list_every_t:
        mean_std_dict[every_t] = {}
        for mu_exponential in list_mu_exponential:
            print(results_dict)
            mean_std_dict[every_t][mu_exponential] = [np.round(np.mean(results_dict[(every_t)][(mu_exponential)]),2),
                                                      np.round(np.std(results_dict[(every_t)][(mu_exponential)]),2)]
    filename = f"mean_std_{algo}_T-{T}_K-{K}_multiple_lambda_mu_n_exps{n_exps}"
    json_path = os.path.join(save_folder, filename + ".json")
    with open(json_path,'w') as f:
        print("save",json_path)
        json.dump(mean_std_dict , f)
    return results_dict




## PARAMS
save_folder = "../results_plots/paper_5_2_dynamic_leaving_20"
K = 4
T = 1000000

list_every_t = [1000, 10000]
list_mu_exponential = [500, 1000, 10000]




mu = np.linspace(0.9,0.1,K)
M = K
config = {}
config['M'] = M
config['mu'] = mu
config['horizon'] = T

n_exps = 2

results_dict_sucb = compute_results_dict(env_config=config,
                                        list_every_t=list_every_t,
                                        list_mu_exponential=list_mu_exponential,
                                        algo="Rnd-SelfishKLUCB",
                                        n_exps=n_exps,
                                        save_folder=save_folder)

results_dict_MC = compute_results_dict(env_config=config,
                                        list_every_t=list_every_t,
                                        list_mu_exponential=list_mu_exponential,
                                        algo="DYN-MC",
                                        n_exps=n_exps,
                                        save_folder=save_folder)
