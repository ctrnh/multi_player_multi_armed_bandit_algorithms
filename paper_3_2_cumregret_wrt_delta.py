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


save_folder = "../results_plots/paper_3_2_cumregret_wrt_delta"

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
delta_dict_given_list = {
    9 :  [
        list(np.round(np.linspace(0.99,.9  ,5),3)) + (list(np.round(np.linspace(0.8,0.7,4),3))) ,
         list(np.round(np.linspace(0.99,0.85 ,5),3)) + (list(np.round(np.linspace(0.8,0.7,4),3))) ,
         list(np.round(np.linspace(0.99,0.81 ,5),3)) + (list(np.round(np.linspace(0.8,0.7,4),3))) ,
         list(np.round(np.linspace(0.99,0.805,5),3)) + (list(np.round(np.linspace(0.8,0.7,4),3))) ,
         list(np.round(np.linspace(0.99,0.801,5),3)) + (list(np.round(np.linspace(0.8,0.7,4),3))) ,
        ]
}

Delta_M5_K_T = {
         9:2000000,
}

mult_sim = MultipleSimulations(M=M,
                 dict_K_T=Delta_M5_K_T,
                dict_given_list=delta_dict_given_list
                              )
mult_sim.run_save_fig(n_exps=n_exps,
                  list_algo_to_plot=algo_to_plot,
                  skip_existing_sim=True)


plot_regret_wrt_xaxis(M=M, K=K, T=Delta_M5_K_T[K],
                        list_algo=algo_to_plot,
                        list_of_mu_list=delta_dict_given_list[K],
                        save_folder=save_folder,
                        wrt="Delta")
