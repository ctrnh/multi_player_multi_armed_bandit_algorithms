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


save_folder = "../results_plots/paper_5_1_dynamic_quasi_asynchronicity"

algo_to_plot = [
                #"Lugosi-Mehrabian 1"
                #"Lugosi-Mehrabian 2",
                #"SIC-MMAB2",
                #"EC-SIC",
                #"SelfishKLUCB",
                #"Bubeck-Budzinski",
                "Rnd-SelfishKLUCB",
                "DYN-MMAB"
                #"MCTopM",
                #"SIC-MMAB",M5_dict_given_list
                ]

n_exps= 50
K = 4
mu = np.linspace(0.9,0.1,K)
M = K


M4_dict_given_list = {
    4: [mu]
}
dict_K_T = {4:500000}

every_t = 10000

dynamic_params = {"can_leave":False,
                  "t_entries":None,
                  "t_leaving":None,
                  "lambda_poisson":1/every_t,
                  "mu_exponential":None}



mult_sim = MultipleSimulations(M=M,
             dict_K_T=dict_K_T,
            dict_given_list=M4_dict_given_list,
            dynamic_params=dynamic_params
                          )

mult_sim.run_save_fig(n_exps=n_exps,
                  list_algo_to_plot=algo_to_plot,
                  skip_existing_sim=True)


save_comparison_dict_mu(M=M,
                   dict_mu=M4_dict_given_list,
                   dict_K_T=dict_K_T,
                   list_algo=algo_to_plot,
                  comparison_folder=save_folder,
                   save_in_separate_folder=True,
                      )

##### 2nd plot: only Rnd-SelfishKLUCB with fixed arrivals (sampled from poisson)

algo_to_plot = [
                #"Lugosi-Mehrabian 1"
                #"Lugosi-Mehrabian 2",
                #"SIC-MMAB2",
                #"EC-SIC",
                #"SelfishKLUCB",
                #"Bubeck-Budzinski",
                "Rnd-SelfishKLUCB",
                #"DYN-MMAB"
                #"MCTopM",
                #"SIC-MMAB",M5_dict_given_list
                ]

n_exps=50
K = 5
mu = np.linspace(0.9,0.1,K)
M = K


M5_dict_given_list = {
    5: [mu]
}
dict_K_T = {5:40000}

every_t = 10000

dynamic_params = {"can_leave":False,
                  "t_entries":[    0,  4912, 13703, 15970, 18278],
                  "t_leaving":None,
                  "lambda_poisson":1/every_t,
                  "mu_exponential":None}



mult_sim = MultipleSimulations(M=M,
             dict_K_T=dict_K_T,
            dict_given_list=M5_dict_given_list,
            dynamic_params=dynamic_params
                          )

mult_sim.run_save_fig(n_exps=n_exps,
                  list_algo_to_plot=algo_to_plot,
                  skip_existing_sim=True)


save_comparison_dict_mu(M=M,
                   dict_mu=M5_dict_given_list,
                   dict_K_T=dict_K_T,
                   list_algo=algo_to_plot,
                  comparison_folder=save_folder,
                   save_in_separate_folder=True,
                      )
