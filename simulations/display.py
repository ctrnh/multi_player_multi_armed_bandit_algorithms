import numpy as np
import matplotlib.pyplot as plt

import collections
import os

import logging
import pickle
import json
import time
import matplotlib as mpl
from environment import *
from simulations import *
from algorithms import *
from tqdm import tqdm
import re

logger = logging.getLogger(__name__)


RESULTS_PATH = "../results"



ALGO_COLORS = {
                "Lugosi-Mehrabian 1":"C3",
                "Lugosi-Mehrabian 2":"C0",
                "SIC-MMAB2":"C2",
                "EC-SIC": "C4",
                "SelfishKLUCB":"C3",
                "Bubeck-Budzinski":"C5",
                "Rnd-SelfishKLUCB":"C1",

                "MCTopM":"C0",
                "SIC-MMAB":"C2",

                "DYN-MMAB": "C2",

                    }
def folder_path_from_env(M,
                         K,
                         mu,):
    return os.path.join(RESULTS_PATH, f"M_{M}-K_{K}", mu_str(mu))

def mu_str(mu):
    mu = sorted(mu,reverse=True)
    return f"mu_{','.join(str(np.round(mu,3)).split())}"

def read_histories(algo_folder,
                   file_type,
                   T,
                   pas=50,
                   percentile=None):
    """
    Args:
        - algo_folder:
            algorithm folder where pickle files are saved
        - percentile:
            if int, then lower/upper represent the x-th percentile. If None, std.
        - file_type:
            "regret" or "collision"
    """
    file_paths = [f for f in os.listdir(algo_folder) if f.startswith(file_type+f'-T_{T}-') and f.endswith(".pkl")]
    whole_arr = []
    for file in file_paths:
        histories = pickle.load(open(os.path.join(algo_folder,file), 'rb'))
        whole_arr += histories

    tarray = [t for t in range(T)][::pas]
    whole_arr = np.array(whole_arr)

    mean = np.mean(whole_arr, axis=0)
    if not percentile:
        std = np.std(whole_arr, axis=0)
        lower = mean - std
        upper = mean + std
    else:
        lower = np.percentile(whole_arr, percentile, axis=0)
        upper = np.percentile(whole_arr, 100 - percentile, axis=0)

    last_values = whole_arr[:,-1]
    n_exps = whole_arr.shape[0]
    return lower, upper, mean, last_values, n_exps, tarray


def plot_regret_wrt_xaxis(M, K, T,
                        list_algo,
                        list_of_mu_list,
                        save_folder,
                        wrt="Delta"):
    """
    Args:
        - wrt (str):
            "Delta" or "mu(K)"
        - list_algo (list of str):
            List of names of algorithms (str(algo))
        - save_folder (path):
            path where image should be saved
    """
    mapping_algo_names = {} # mapping from saving_name to algo_name
    for algo_name in list_algo:
        mapping_algo_names["".join(re.split(r"-| |_",algo_name)).lower()] = algo_name


    title = f"M = {M}, K = {K}"
    for file_type in [
        "regret",
        #"collision"
        ]:
        plt.figure(figsize=(12,7))
        y_label = f"Total cumulative {file_type}"
        x_label = wrt
        all_algo_str = "_".join(list(mapping_algo_names.keys()))
        fig_name = f"M_{M}-K_{K}-T_{T}_{file_type}_wrt_{wrt}_{all_algo_str}.eps"


        mean_stds = {}
        for mu_list in list_of_mu_list:

            env_folder_path = folder_path_from_env(M=M,
                                 K=K,
                                 mu=mu_list)
            #print(mu_list)
            for f in os.listdir(env_folder_path):
                if f in mapping_algo_names:
                    algo_folder = os.path.join(env_folder_path, f)
                    lower, upper, mean, last_values, n_exps, tarray = read_histories(algo_folder=algo_folder,
                                                                                       file_type=file_type,
                                                                                       T=T,
                                                                                       pas=50,
                                                                                       percentile=False)
                    if f not in mean_stds:
                        mean_stds[f] = {"means":[], "stds":[]}
                    mean_stds[f]["means"].append(mean[-1])
                    mean_stds[f]["stds"].append(upper[-1]-mean[-1])
                    #print(f," n_exps:", n_exps)

        x_axis = []
        for mu in list_of_mu_list:
            mu = sorted(mu,reverse=True)
            if wrt == "Delta":
                x_axis.append(mu[M-1] - mu[M])
            elif wrt == "mu(K)":
                x_axis.append(mu[-1])
            else:
                raise NotImplementedError

        for algo in mean_stds.keys():
            plt.errorbar(x_axis,mean_stds[algo]["means"], yerr=mean_stds[algo]["stds"],fmt='.--',lw=0.8, capsize=3,
                        label=mapping_algo_names[algo],
                        color=ALGO_COLORS[mapping_algo_names[algo]])
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        #plt.title(title)
        plt.legend()
        plt.grid()
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        print("Saving figure to ", os.path.join(save_folder, fig_name))
        plt.savefig(os.path.join(save_folder, fig_name),format="eps")



def plot_regret_wrt_M(mu, K, T,
                        list_algo,
                        list_of_M,
                        save_folder,
                        ):

    mapping_algo_names = {} # mapping from saving_name to algo_name
    for algo_name in list_algo:
        mapping_algo_names["".join(re.split(r"-| |_",algo_name)).lower()] = algo_name

    wrt = "M"
    for file_type in [
        "regret",
        #"collision"
        ]:
        plt.figure(figsize=(12,7))
        y_label = f"Last cumulative {file_type}"
        x_label = wrt
        all_algo_str = "_".join(list(mapping_algo_names.keys()))

        title = f"K = {K}"
        fig_name = f"{mu_str(mu)}_K_{K}-T_{T}_{file_type}_wrt_{wrt}_{all_algo_str}.eps"


        mean_stds = {}
        for M in list_of_M:
            env_folder_path = folder_path_from_env(M=M,
                                 K=K,
                                 mu=mu)
            #print(mu_list)
            for f in os.listdir(env_folder_path):
                if f in mapping_algo_names:
                    algo_folder = os.path.join(env_folder_path, f)
                    lower, upper, mean, last_values, n_exps, tarray = read_histories(algo_folder=algo_folder,
                                                                                   file_type=file_type,
                                                                                   T=T,
                                                                                   pas=50,
                                                                                   percentile=False)
                    if f not in mean_stds:
                        mean_stds[f] = {"means":[], "stds":[]}
                    mean_stds[f]["means"].append(mean[-1])
                    mean_stds[f]["stds"].append(upper[-1]-mean[-1])
                    #print(f," n_exps:", n_exps)

        x_axis = list_of_M

        for algo in mean_stds.keys():
            plt.errorbar(x_axis,mean_stds[algo]["means"],
                        yerr=mean_stds[algo]["stds"],fmt='.--',lw=0.8,
                        capsize=3, label=mapping_algo_names[algo],
                        color=ALGO_COLORS[mapping_algo_names[algo]])
        plt.ylabel(y_label)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        plt.xlabel(x_label)
        #plt.title(title)
        plt.legend()
        plt.grid()
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        print("Saving figure to ", os.path.join(save_folder, fig_name))
        plt.savefig(os.path.join(save_folder, fig_name),format="eps")




def save_comparison_dict_mu(M,
                            dict_mu,
                            dict_K_T,
                            list_algo,
                           comparison_folder,
                            save_in_separate_folder=False,
                            percentile=False,
                            plot_histogram=False):
    """
    Args:
        - list_algo:
            List of algorithm names
        - comparison_folder (path):
            path where comparison plots are to be saved
        - save_in_separate folder (bool):
            If True plots will be saved like: comparison_folder/M_4-K-5/mu_[..]/....png
    """
    for K in dict_K_T:
        T = dict_K_T[K]
        if not K in dict_mu:
            break
        list_mu = dict_mu[K]
        for mu in list_mu:
            mu = sorted(mu,reverse=True)
            mu_folder = os.path.join(RESULTS_PATH, f"M_{M}-K_{K}", mu_str(mu))
            if save_in_separate_folder:
                direct_folder_of_plots = os.path.join(comparison_folder, f"M_{M}-K_{K}")#, mu_str(mu))
                if not os.path.exists(direct_folder_of_plots):
                    os.makedirs(direct_folder_of_plots)
            save_comparison_figures(mu_folder,
                            T,
                            list_algo,
                            comparison_folder=direct_folder_of_plots,
                            add_to_file_path=mu_str(mu),
                            percentile=percentile,
                            plot_histogram=plot_histogram
                           )

def save_comparison_figures(mu_folder,
                            T,
                            list_algo,
                            comparison_folder=None,
                            add_to_file_path="",
                            plot_histogram=False,
                            percentile=False
                           ):
    """
    Args:
        - list_algo:
            List of algorithm names
    save comparison figure for all algo of list_algo, in the corresponding environment
    save into mu_folder/comparison
    from pickle files.
    """
    mapping_algo_names = {} # mapping from saving_name to algo_name
    for algo_name in list_algo:
        mapping_algo_names["".join(re.split(r"-| |_",algo_name)).lower()] = algo_name


    fig_type=['regret',
              #'collision'
             ]
    mu = eval(mu_folder.split('_')[-1])
    if comparison_folder is None:
        comparison_folder = os.path.join(mu_folder, "comparison")
    if not os.path.exists(comparison_folder):
        os.makedirs(comparison_folder)

    new_list_algo = []
    for file_type in fig_type:
        if plot_histogram:
            fh, ax_histogram = plt.subplots()
        fcum, ax_cum = plt.subplots(figsize=(12,7))

        for algo in list(mapping_algo_names.keys()):
            algo_folder = os.path.join(mu_folder, algo)
            if os.path.exists(algo_folder):
                lower, upper, mean, last_values, n_exps, tarray = read_histories(algo_folder,
                                                                       file_type=file_type,
                                                                       T=T,
                                                                       percentile=percentile)

                if plot_histogram:
                    ax_histogram.hist(last_values, alpha=0.5, label=mapping_algo_names[algo])

                ax_cum.fill_between(tarray, lower, upper,alpha=0.2,color=ALGO_COLORS[mapping_algo_names[algo]])
                ax_cum.plot(tarray, mean, label=mapping_algo_names[algo],color=ALGO_COLORS[mapping_algo_names[algo]])
                if algo not in new_list_algo:
                    new_list_algo.append(algo)
            else:
                print(f"{algo_folder} does not exist")

        all_algo_name = "_".join(new_list_algo)
        cum_path = os.path.join(comparison_folder, f"{add_to_file_path}{all_algo_name}-cum{file_type}-T_{T}-n_exps_{n_exps}.png")
        ax_cum.set_xlabel('t')
        ax_cum.set_ylabel('Cumulative ' + file_type)
        ax_cum.grid()
        ax_cum.legend()

        ax_cum.tick_params(axis='both', which='major', labelsize=20)
        ax_cum.ticklabel_format(axis="both", style="sci", scilimits=(0,0))
        #ax_cum.set_title(title)
        fcum.tight_layout()
        fcum.savefig(cum_path)

        if plot_histogram:
            histogram_path = os.path.join(comparison_folder, f"{all_algo_name}-hist{file_type}-T_{T}-n_exps_{n_exps}.png")
            ax_histogram.set_xlabel('Total cumulative regret')
            ax_histogram.set_ylabel('Number of runs')
            ax_histogram.legend()
            #ax_histogram.set_title(title)
            fh.tight_layout()
            fh.savefig(histogram_path)
            plt.close(fh)
            plt.close(fcum)
