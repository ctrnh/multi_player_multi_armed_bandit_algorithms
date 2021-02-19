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

import argparse
#
# parser = argparse.ArgumentParser()
# parser.add_argument("--run_simulations", type=str, help="2510", default="2510")
# parser.add_argument("--save_plots", type=bool, default="1")
# parser.add_argument("--n_exps", type=int)
# args = parser.parse_args()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


n_exps =50
algo_to_plot = [
                #"Lugosi-Mehrabian 1"
                #"Lugosi-Mehrabian 2",
                #"SIC-MMAB2",
                #"EC-SIC",
                #"SelfishKLUCB",
                #"Bubeck-Budzinski",
                "Rnd-SelfishKLUCB",
                "MCTopM",
                "SIC-MMAB",

                #DYN-MMAB
                ]
comparison_folder = "../results_plots/paper_4_comparison_with_collision"

####

M = 2
M2_sota_dict_given_list = {
    3 : [[0.2,0.15,0.1,],
        [0.9,0.85,0.8,],
        [0.99,0.5,0.01]],
    5: [[0.2,0.175,0.15,0.125,0.1,],
       [0.9,0.875,0.85,0.825,0.8,],
        [0.99,0.745,0.5,0.255,0.01,]
       ],
    20: [[0.9,0.895,0.889,0.884,0.879,0.874,0.868,0.863,0.858,0.853,0.847,0.842,0.837,0.832,0.826,0.821,0.816,0.811,0.805,0.8,]]

}
M2_dict_K_T = {3:500000,
           5:500000,
           20:500000}




################
M = 5
M5_sota_dict_given_list = {
    10: [
        [0.9,0.889,0.878,0.867,0.856,0.844,0.833,0.822,0.811,0.8,],
        [0.2,0.189,0.178,0.167,0.156,0.144,0.133,0.122,0.111,0.1,],
        [0.99,0.881,0.772,0.663,0.554,0.446,0.337,0.228,0.119,0.01,]
       ],

}

M5_dict_K_T = { 10:2000000}




#####################
M = 10
M10_sota_dict_given_list = {
    11 : [
        #[0.2,0.19,0.18,0.17,0.16,0.15,0.14,0.13,0.12,0.11,0.1,],
        #[0.9,0.89,0.88,0.87,0.86,0.85,0.84,0.83,0.82,0.81,0.8,],
        #[0.99,0.892,0.794,0.696,0.598,0.5,0.402,0.304,0.206,0.108,0.01,]
        ],

    15: [
        [0.2,0.193,0.186,0.179,0.171,0.164,0.157,0.15,0.143,0.136,0.129,0.121,0.114,0.107,0.1,],
        [0.9,0.893,0.886,0.879,0.871,0.864,0.857,0.85,0.843,0.836,0.829,0.821,0.814,0.807,0.8,],
        [0.99,0.92,0.85,0.78,0.71,0.64,0.57,0.5,0.43,0.36,0.29,0.22,0.15,0.08,0.01]
       ],

}

M10_dict_K_T = {11:2000000,15:2000000}







mult_sim = MultipleSimulations(M=2,
                 dict_K_T=M2_dict_K_T,
                dict_given_list=M2_sota_dict_given_list
                              )

mult_sim.run_save_fig(n_exps=n_exps,
                      list_algo_to_plot=algo_to_plot,
                      skip_existing_sim=True)

print("Running for M= 5")
mult_sim = MultipleSimulations(M=5,
             dict_K_T=M5_dict_K_T,
            dict_given_list=M5_sota_dict_given_list
                         )
mult_sim.run_save_fig(n_exps=n_exps,
                  list_algo_to_plot=algo_to_plot,
                  skip_existing_sim=True)

print("Running for M= 10")
mult_sim = MultipleSimulations(M=10,
                 dict_K_T=M10_dict_K_T,
                dict_given_list=M10_sota_dict_given_list
                              )
mult_sim.run_save_fig(n_exps=n_exps,
                  list_algo_to_plot=algo_to_plot,
                  skip_existing_sim=True)

save_comparison_dict_mu(M=2,
                    dict_mu=M2_sota_dict_given_list,
                    dict_K_T=M2_dict_K_T,
                    list_algo=algo_to_plot,
                   comparison_folder=comparison_folder,
                    save_in_separate_folder=True,
                   )

save_comparison_dict_mu(M=5,
                    dict_mu=M5_sota_dict_given_list,
                    dict_K_T=M5_dict_K_T,
                    list_algo=algo_to_plot,
                   comparison_folder=comparison_folder,
                    save_in_separate_folder=True,
                   )

save_comparison_dict_mu(M=10,
                   dict_mu=M10_sota_dict_given_list,
                   dict_K_T=M10_dict_K_T,
                   list_algo=algo_to_plot,
                  comparison_folder=comparison_folder,
                   save_in_separate_folder=True,
                      )
