import numpy as np
import matplotlib.pyplot as plt

import collections
import os

import logging
import pickle
import json
import time



from tqdm import tqdm

from environment import *
from simulations import *
from algorithms import *
from simulations import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


comparison_folder = "../results_plots/paper_3_3_corner_case"

M = 5
n_exps = 50
algo_to_plot = [
                #"Lugosi-Mehrabian 1"
                "Lugosi-Mehrabian 2",
                "SIC-MMAB2",
                #"EC-SIC",
                #"SelfishKLUCB",
                #"Bubeck-Budzinski",
                "Rnd-SelfishKLUCB",

                #"MCTopM",
                #"SIC-MMAB",
                ]

M5_dict_given_list = {
    10: [
        [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5],
        [0.509,0.507,0.506,0.505,0.505,0.502,0.5,0.498,0.492,0.492], # Generated from a uniform U([0.49,0.51]) distribution
        ]
}

dict_K_T = {10:2000000}
mult_sim = MultipleSimulations(M=M,
             dict_K_T=dict_K_T,
            dict_given_list=M5_dict_given_list
                          )
mult_sim.run_save_fig(n_exps=n_exps,
                  list_algo_to_plot=algo_to_plot,
                  skip_existing_sim=True)

save_comparison_dict_mu(M=M,
                   dict_mu=M5_dict_given_list,
                   dict_K_T=dict_K_T,
                   list_algo=algo_to_plot,
                  comparison_folder=comparison_folder,
                   save_in_separate_folder=True,
                      )
