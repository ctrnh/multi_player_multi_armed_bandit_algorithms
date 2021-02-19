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

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

comparison_folder = "../results_plots/paper_2_2_rndsucb_sucb"


M = 2
n_exps = 2
algo_to_plot = [
                #"Lugosi-Mehrabian 1"
                #"Lugosi-Mehrabian 2",
                #"SIC-MMAB2",
                #"EC-SIC",
                "SelfishKLUCB",
                #"Bubeck-Budzinski",
                "Rnd-SelfishKLUCB",

                #"MCTopM",
                #"SIC-MMAB",

                #DYN-MMAB
                ]
M2_sota_dict_given_list = {
    2 : [[0.1,0.9,]]

}
dict_K_T = {2:10000,}


mult_sim = MultipleSimulations(M=M,
                dict_K_T=dict_K_T,
                dict_given_list=M2_sota_dict_given_list
                              )
mult_sim.run_save_fig(n_exps=n_exps,
                      list_algo_to_plot=algo_to_plot,
                      skip_existing_sim=True)

save_comparison_dict_mu(M=M,
                    dict_mu=M2_sota_dict_given_list,
                    dict_K_T=dict_K_T,
                    list_algo=algo_to_plot,
                   comparison_folder=comparison_folder,
                    save_in_separate_folder=True,
                    percentile=10,
                    plot_histogram=True
                   )
