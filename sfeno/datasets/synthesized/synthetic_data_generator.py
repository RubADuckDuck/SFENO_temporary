import random
import networkx as nx

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint


def create_random_sparse_adjmat(n, n_in, n_out):
    mat = np.zeros((n, n))
    for i in range(n):
        random_idxes = random.choices([j for j in range(n)], k=n_in)  # get n_in number of random numbers from 0~n

        for r_idx in random_idxes:
            mat[i, r_idx] = 1

    for i in range(n):
        random_idxes = random.choices([j for j in range(n)], k=n_in)  # get n_in number of random numbers from 0~n

        for r_idx in random_idxes:
            mat[r_idx, i] = 1

    return mat


class RandomExpressionPerturbationGenerator:
    def __init__(self, n, F=None):
        self.n = n

        self.F = F

        self.n_in = 1
        self.n_out = 1

        self.pert_range = 10

        self.dataset = None

        self.dt = 0.1
        self.nt = 400

    def create_datasets(self):

        # create graph
        adj_mat = self.create_random_sparse_matrix()

        # create W with graph
        weight_matrix = np.random.randn(self.n, self.n)
        weight_matrix = adj_mat * weight_matrix

        # create epsilon
        epsilon = np.abs(np.random.randn(self.n))

        # create alpha
        alpha = np.abs(np.random.randn(self.n))

        # get perturbations
        perts = self.pert_gen()

        # expression
        expr = np.zeros_like(perts)

        if self.F is None:
            F = lambda s, t, mu: epsilon * np.tanh(np.matmul(weight_matrix, s) + mu) - alpha * s
        else:
            F = self.F

        # get result expressions for each perturbation
        for i in range(len(perts)):
            expr[i] = self.solve_ode(F, perts[i])

        return perts, expr

    def solve_ode(self, F, pert):
        y0 = pert
        mu = pert

        t = np.array([i * self.dt for i in range(self.nt)])

        # solve ode and return result expression
        sol = odeint(F, y0, t, args=(mu,))

        # check convergence

        return sol[-1, :]

    def pert_gen(self, n_limit=None):
        # int to binary
        get_bin = lambda x, n: format(x, 'b').zfill(n)

        n_perturbation = 2 ** self.pert_range

        pert_mat = np.zeros((n_perturbation, self.n))

        for i in range(n_perturbation):
            temp = list(get_bin(i, self.n))
            temp = np.array(list(map(int, temp)))

            pert_mat[i] = temp

        if n_limit is None:
            return pert_mat

    def create_random_sparse_matrix(self):
        return create_random_sparse_adjmat(self.n, self.n_in, self.n_out)

    def get_ode_result(self, w_mat):
        # get W and solve ode to return steady state
        return None