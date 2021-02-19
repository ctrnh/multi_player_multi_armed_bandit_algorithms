cimport cython
import numpy as np
cimport numpy as np
import sys

from libc.math cimport log as c_log

ctypedef np.float64_t DOUBLE
ctypedef np.int32_t INT

np.import_array()


cdef inline double double_max(double a, double b): return a if a >= b else b
cdef inline double double_min(double a, double b): return a if a <= b else b


cdef double klBern(double x, double y):
    cdef:
        double eps = 1e-7
    x = double_min(double_max(x, 1e-7), 1-1e-7)
    y = double_min(double_max(y, 1e-7), 1-1e-7)
    return x*c_log(x/y) + (1-x)*c_log((1-x)/(1-y))

cpdef np.ndarray[DOUBLE, ndim=2] computeKLUCB(
        int t,
        np.ndarray[DOUBLE, ndim=2] mu_hat,
        np.ndarray[INT, ndim=2] pulls,):

    cdef:
        Py_ssize_t player
        Py_ssize_t arm
        double eps = 1e-7
        double precision = 1e-3
        double d = 0
        double u = 1
        double m = 0
        double l = 0
        int count = 0
        cdef Py_ssize_t M = mu_hat.shape[0]
        cdef Py_ssize_t K = mu_hat.shape[1]
        np.ndarray[DOUBLE, ndim=2] ucb_idx = np.zeros(
                (M,K), dtype=np.float64)

    for player in range(M):
        for arm in range(K):
            d = (c_log(t)+3*c_log(c_log(t))) / (eps +pulls[player,arm])

            u = 1
            l = mu_hat[player, arm]
            count = 0
            while u-l > precision and count <= 50:
                count += 1
                m = (l+u)/2
                if klBern(mu_hat[player,arm], m) > d:
                    u = m
                else:
                    l = m
            ucb_idx[player, arm] = (l+u)/2
    return ucb_idx
