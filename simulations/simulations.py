import numpy as np
import matplotlib.pyplot as plt

import collections
import os

import logging
import pickle
import json
import time
logger = logging.getLogger()



from tqdm import tqdm, tqdm_notebook
import importlib

from environment import *
from simulations import *
from algorithms import *
import re

class Simulation:
    """
    Helper class for a given single environment + a list of algorithms to run in this environment
    - Runs and save algorithms results in an organized way in '../results', which is created if it doesn't exist already
    """
    def __init__(self,
                 environment,
                 list_algo,
                skip_existing_sim=True):
        logger.info(f"skip existing sim set to {skip_existing_sim}")
        self.skip_existing_sim = skip_existing_sim
        self.list_algo = list_algo
        self.env = environment

        self.hist_folder = os.path.join("../results",
                                  f"M_{self.env.M}-K_{self.env.K}",
                                  f"mu_{','.join(str(np.round(self.env.mu,3)).split())}"
                                     )
        self.figure_folder = self.hist_folder
        for folder in [self.hist_folder, self.figure_folder]:
            for algo in list_algo:
                algo_folder = os.path.join(folder, self.saving_name(algo))
                if not os.path.exists(algo_folder):
                    os.makedirs(algo_folder)
                    logging.info(f'Creating folder {algo_folder}')

    def saving_name(self,algo):
        """
        Given an algorithm, returns a string name which is better formated for saving
        e.g. : algorithm Lugosi2 which str is "Lugosi 2" becomes "lugosi2"
        """
        return "".join(re.split(r"-| |_",str(algo))).lower()

    def read_histories(self, algo, file_type, percentile):
        """
        Reads pickle files of a given
        Returns:
            - lower/upper: (array of dim 1) lower/upper confidence interval of cumulative regret/collision
            - mean: (array of dim 1) average over runs of cumulative regret/collision
            - last_values: total cumulative regrets/collision
            - n_exps: total number of runs
            - tarray: (array of dim 1)
        """
        algo_folder = os.path.join(self.hist_folder, self.saving_name(algo))
        file_paths = [f for f in os.listdir(algo_folder) if f.startswith(file_type+f'-T_{self.env.T}-')]
        whole_arr = []

        for file in file_paths:
            histories = pickle.load(open(os.path.join(algo_folder,file), 'rb'))
            whole_arr += histories

        pas = 50
        tarray = [t for t in range(self.env.T)][::pas]

        whole_arr = np.array(whole_arr)

        mean = np.mean(whole_arr, axis=0)
        if percentile is None:
            std = np.std(whole_arr, axis=0)
            lower = mean - std
            upper = mean + std
        else:
            lower = np.percentile(whole_arr, percentile, axis=0)
            upper = np.percentile(whole_arr, 100 - percentile, axis=0)

        last_values = whole_arr[:,-1]
        n_exps = whole_arr.shape[0]
        return lower, upper, mean, last_values, n_exps, tarray

    def save_individual_figure(self, algo, file_type='regret', save_what='histogram_wrt_t', percentile=10):
        lower, upper, mean, last_values, n_exps,tarray = self.read_histories(algo=algo, file_type=file_type, percentile=percentile)
        fig_algo_folder = os.path.join(self.figure_folder, self.saving_name(algo))
        if "histogram" in save_what:
            plt.figure()
            tmp = plt.hist(last_values)
            plt.title("Histogram of last cumulative " + file_type +", "+ str(n_exps) + " exps")
            fig_path = os.path.join(fig_algo_folder, "histogram-"+file_type + f"-T_{self.env.T}" + "-n_exps_" +str(n_exps)+ ".eps")
            plt.savefig(fig_path, format="eps")
            plt.close()
        if "wrt_t" in save_what:
            plt.figure()
            plt.fill_between(tarray, lower, upper,alpha=0.2)
            plt.plot(tarray, mean)
            plt.xlabel('t')
            plt.ylabel('Cumulative ' + file_type)
            plt.grid()
            plt.tight_layout()
            fig_path = os.path.join(fig_algo_folder, "cum" + file_type + f"-T_{self.env.T}" + "-n_exps_" +str(n_exps)+ ".png")
            plt.savefig(fig_path, format="png")
            plt.close()


    def save_individual_figures_all(self, percentile):
        for algo in self.list_algo:
            for file_type in ['regret', 'collision']:
                self.save_individual_figure(algo=algo, file_type=file_type, percentile=percentile)


    def algo_filenames(self, algo, nb_exps,pas):
        filenames = []
        for file_type in ["regret","collision"]:
            idx = 0
            file = os.path.join(self.hist_folder,
                                self.saving_name(algo),
                                f"{file_type}-T_{self.env.T}-pas_{pas:03}-n_exps_{nb_exps}-{idx:02}.pkl")
            if os.path.exists(file) and self.skip_existing_sim:
                logger.info(f"{file} already exists so skip")
                return (None,None)
            while os.path.exists(file) and idx < 100 :
                file = os.path.join(self.hist_folder,
                                    self.saving_name(algo),
                                    f"{file_type}-T_{self.env.T}-pas_{pas:03}-n_exps_{nb_exps}-{idx:02}.pkl")

                idx += 1
            filenames.append(file)
        return filenames

    def run_save_hist(self,
                nb_exps,
                pas=50,
               save_regret=True,
               save_collisions=True
               ):
        """
        run algos in list_algo + save regrets and collisions into .pkl files
        """
        for algo in self.list_algo:
            logger.info(f"Running {algo}")

            algo_regrets = []
            algo_collision_sum = []
            algo_regret_file, algo_collision_file = self.algo_filenames(algo, nb_exps=nb_exps, pas=pas)
            if algo_regret_file is not None:
                for i_exp in tqdm(range(nb_exps)):
                    algo.run()
                    algo_regrets.append(list(np.cumsum(algo.regret))[::pas])
                    algo_collision_sum.append(list(np.cumsum(np.sum(algo.collision_hist, axis=0)))[::pas]) # sum of collisions (over arms) not cum coll wrt t
                    algo.reset()
                if save_regret:
                    with open(algo_regret_file,'wb') as f:
                        logging.info(f"Saving regrets at {algo_regret_file}" )
                        pickle.dump(algo_regrets , f)
                if save_collisions:
                    with open(algo_collision_file,'wb') as f:
                        logging.info(f"Saving collisions at {algo_collision_file}")
                        pickle.dump(algo_collision_sum , f)
