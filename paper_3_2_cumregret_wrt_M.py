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


save_folder = "../results_plots/paper_3_2_cumregret_wrt_M"

n_exps = 50
algo_to_plot = [
                #"Lugosi-Mehrabian 1"
                "Lugosi-Mehrabian 2",
                "SIC-MMAB2",
                "EC-SIC",
                #"SelfishKLUCB",
                #"Bubeck-Budzinski",
                "Rnd-SelfishKLUCB",

                #"MCTopM",
                #"SIC-MMAB",
                ]

K = 10
mu =  list(np.round(np.linspace(0.9,0.1,K),3))
wrt_M_dict_given_list = {10:[mu]}
dict_K_T = {10: 2000000}


for M in range(1,K):
    mult_sim = MultipleSimulations(M=M,
                 dict_K_T=dict_K_T,
                dict_given_list=wrt_M_dict_given_list
                              )
    mult_sim.run_save_fig(n_exps=n_exps,
                      list_algo_to_plot=algo_to_plot,
                      skip_existing_sim=True)



## DISPLAY
plot_regret_wrt_M(mu=mu, K=K, T=dict_K_T[K],
                    list_algo=algo_to_plot,
                    list_of_M=[i for i in range(1,10)],
                    save_folder=save_folder,
                )
