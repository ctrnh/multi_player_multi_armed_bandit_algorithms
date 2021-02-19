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

parser = argparse.ArgumentParser()
parser.add_argument("--run_simulations", type=bool, help="1 if run simulation", default="1")
parser.add_argument("--save_plots", type=bool, default="1")
args = parser.parse_args()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


n_exps = 30
algo_to_plot = [
                #"Lugosi-Mehrabian 1"
                #"Lugosi-Mehrabian 2",
                #"SIC-MMAB2",
                #"EC-SIC",
                #"SelfishKLUCB",
                "Bubeck-Budzinski",
                #"Rnd-SelfishKLUCB",

                #"MCTopM",
                #"SIC-MMAB",

                #DYN-MMAB
                ]
comparison_folder = "../results_plots/paper_3_1_bubeck"

####

M = 2
M2_sota_dict_given_list = {
    3 : [[0.2,0.15,0.1,],
        [0.9,0.85,0.8,],
        [0.99,0.5,0.01]],

}
M2_dict_K_T = {3:500000,
           #5:100000,
           #20:100000
           }







if args.run_simulations:
    mult_sim = MultipleSimulations(M=2,
                     dict_K_T=M2_dict_K_T,
                    dict_given_list=M2_sota_dict_given_list
                                  )

    mult_sim.run_save_fig(n_exps=n_exps,
                          list_algo_to_plot=algo_to_plot,
                          skip_existing_sim=True)

    # mult_sim = MultipleSimulations(M=5,
    #              dict_K_T=M5_dict_K_T,
    #             dict_given_list=M5_sota_dict_given_list
    #                           )
    # mult_sim.run_save_fig(n_exps=n_exps,
    #                   list_algo_to_plot=algo_to_plot,
    #                   skip_existing_sim=True)
    #
    # mult_sim = MultipleSimulations(M=10,
    #                  dict_K_T=M10_dict_K_T,
    #                 dict_given_list=M10_sota_dict_given_list
    #                               )
    # mult_sim.run_save_fig(n_exps=n_exps,
    #                   list_algo_to_plot=algo_to_plot,
    #                   skip_existing_sim=True)
if args.save_plots:
    save_comparison_dict_mu(M=2,
                        dict_mu=M2_sota_dict_given_list,
                        dict_K_T=M2_dict_K_T,
                        list_algo=algo_to_plot,
                       comparison_folder=comparison_folder,
                        save_in_separate_folder=True,
                       )

    # save_comparison_dict_mu(M=5,
    #                     dict_mu=M5_sota_dict_given_list,
    #                     dict_K_T=M5_dict_K_T,
    #                     list_algo=algo_to_plot,
    #                    comparison_folder=comparison_folder,
    #                     save_in_separate_folder=True,
    #                    )
    #
    # save_comparison_dict_mu(M=10,
    #                    dict_mu=M10_sota_dict_given_list,
    #                    dict_K_T=M10_dict_K_T,
    #                    list_algo=algo_to_plot,
    #                   comparison_folder=comparison_folder,
    #                    save_in_separate_folder=True,
    #                       )
