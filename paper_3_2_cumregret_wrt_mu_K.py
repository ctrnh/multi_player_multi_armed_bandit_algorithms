import numpy as np
import matplotlib.pyplot as plt

import collections
import os

import logging
import pickle
import json
import time



from tqdm import tqdm, tqdm_notebook
import importlib

from environment import *
from simulations import *
from algorithms import *
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

save_folder = "../results_plots/paper_3_2_cumregret_wrt_mu_K"

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

M = 5
K = 9
mu_min_dict_given_list = {
    9 : [list(np.round(np.linspace(i*0.1,0.9,9),3)) for i in range(1,9)]
}

mu_min_M5_K_T = { #5:800000,
         9:2000000,
}

mult_sim = MultipleSimulations(M=M,
                 dict_K_T=mu_min_M5_K_T,
                dict_given_list=mu_min_dict_given_list
                              )
mult_sim.run_save_fig(n_exps=n_exps,
                  list_algo_to_plot=algo_to_plot,
                  skip_existing_sim=True)


plot_regret_wrt_xaxis(M=M, K=K, T=mu_min_M5_K_T[K],
                        list_algo=algo_to_plot,
                        list_of_mu_list=mu_min_dict_given_list[K],
                        save_folder=save_folder,
                        wrt="mu(K)")
