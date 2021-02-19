import numpy as np
import matplotlib.pyplot as plt

import collections
import os

import logging
import pickle
logger = logging.getLogger()
logger.setLevel(logging.INFO)


from tqdm import tqdm


def best_arm_rnd(my_list):
    """
    singel idx
    returns the index i for which list[i] = max
    breaks ties at random
    """
    my_list = np.array(my_list)
    max_val = np.max(my_list)
    all_max = np.argwhere(my_list == max_val).flatten()
    if len(all_max)== 0:
        return np.random.choice(len(my_list))
    return np.random.choice(all_max)


def best_arm(indices, tie_breaking="random"):
    """
    indices : (M x K)
    returns array (M,) with chosen arms
    """

    if tie_breaking == "random":
        maxi = np.max(indices, axis=1)
        M = indices.shape[0]
        chosen_arms = np.zeros((M,))
        for player in range(M):
            chosen_arms[player] = np.random.choice(np.where(indices[player] == maxi[player])[0])
    elif "lexico" in tie_breaking:
        chosen_arms = np.argmax(indices, axis=1)
    return chosen_arms.astype(int)


def klBern(x, y):
    """Kullback-Leibler divergence for Bernoulli distributions."""
    eps = 1e-10
    x = min(max(x, eps), 1-eps)
    y = min(max(y, eps), 1-eps)
    return x*np.log(x/y) + (1-x)*np.log((1-x)/(1-y))

def klucb(x, d, div, upperbound,
          lowerbound=-float('inf'), precision=1e-2, max_iter=50):
    """The generic klUCB index computation.

    Input args.: x, d, div, upperbound, lowerbound=-float('inf'), precision=1e-6,
    where div is the KL divergence to be used.

    finds argmax_{q \in [x,upperbound]} (div(x,q) <= d)

    """
    l = max(x, lowerbound)
    u = upperbound
    count = 0
    while u-l>precision and count <= max_iter:
        count += 1
        m = (l+u)/2
        if div(x, m) > d:
            u = m
        else:
            l = m
    return (l+u)/2

def klucb_ber(x, d):
    ub = 1
    return klucb(x, d, klBern, ub)
