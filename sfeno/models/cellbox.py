import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sfeno.solvers import get_ode_solver

import math
import pandas as pd

# def factory(args):
#     """Create a model based on configuration"""
#     return CellBox.build_model(args)
#

def get_node_ss_clt(node_part_idxes):
    node_subs_collection = []

    for i, part_idx in enumerate(node_part_idxes):
        if i == 0:
            cur_subset = [j for j in range(0, node_part_idxes[i])]
        else:
            cur_subset = [j for j in range(node_part_idxes[i - 1], node_part_idxes[i])]

        node_subs_collection.append(cur_subset)

    return node_subs_collection

def create_mask(n, node_subs_collection, adj_mat):
    # print(n,node_subs_collection, adj_mat) # debug

    W_mask = torch.zeros((n,n))

    n_subset = len(node_subs_collection)

    for i_in in range(n_subset):
        for j_out in range(n_subset):
            connection = adj_mat[i_in,j_out]

            for i in node_subs_collection[i_in]:
                for j in node_subs_collection[j_out]:
                    W_mask[i,j] = connection

    return W_mask

class Ode(nn.Module):
    # call build model after creating parameters have to be initialized
    def __init__(self):
        super().__init__()

        # parameters of model
        self.W = 0
        self.alpha_bs = 0  # before softmax
        self.epsilon_bs = 0  # befor softmax

        self.W_mask = 0

        self.params = {}

        self.n_x = 0

        # u, strength of perturbation on node is currently fixed as a constant
        # meaning only shows the case where the drug effect is assumed to be constant
        self.u = 0
        # if possible make u as a function when given t returns the strength of perturbation

        self.param_max_min = {
            'W_min': math.inf,
            'W_max': - math.inf,
            'alpha_max': 0,
            'eps_max': 0
        }

    def build_model(self,n_x, node_subs_collection, adj_mat):
        self.n_x = n_x
        self.init_params(n_x, node_subs_collection, adj_mat)
        return

    def init_params(self, n_x, node_subs_collection, adj_mat):
        """
        Initialize parameters in the Hopfield equation
        Enforce constraints  (i: recipient)
           no self regulation wii=0
           ingoing wij for drug nodes (88th to 99th) = 0 [i_pt_drg_activity_nodes 87: ]
                            w [87:99,_] = 0
           outgoing wij for phenotypic nodes (83th to 87th) [i_pt_protein_nodes 82 : i_pt_drg_activity_nodes 87]
                            w [_, 82:87] = 0
           ingoing wij for phenotypic nodes from drug ndoes (direct) [i_pt_protein_nodes 82 : i_pt_drg_activity_nodes 87]
                            w [82:87, 87:99] = 0
        """
        W = torch.Tensor(np.random.normal(loc=0.01, scale=1.0, size=(n_x, n_x)))
        W.requires_grad = True


        W_mask = create_mask(n_x, node_subs_collection, adj_mat)

        W_mask = torch.Tensor(W_mask)
        W_mask.requires_grad = False

        self.W = W
        self.W_mask = W_mask

        eps = torch.tensor(np.ones((n_x, 1)), dtype=torch.float32)
        alpha = torch.tensor(np.ones((n_x, 1)), dtype=torch.float32)
        eps.requires_grad = True # check if True
        alpha.requires_grad = True  # check if True

        self.alpha_bs = alpha # before softmax
        self.epsilon_bs = eps # befor softmax

        self.params['W'] = self.W
        self.params['alpha_bs'] = self.alpha_bs
        self.params['eps_bs'] = self.epsilon_bs

        return

    def get_param_dict(self):
        return self.params

    def perturb(self, u):
        # called to set u differently for each perterbation(data sample)
        self.u = u
        return

    def forward(self, t, x):
        """calculate the derivatives dx/dt in the ODEs"""
        # x is the state
        # u is perturbation strength

        batch, _, _ = x.shape
        u = self.u.reshape(batch, -1, 1)

        eps = F.softplus(self.epsilon_bs)
        alpha = F.softplus(self.alpha_bs)


        masked_W_broadcasted = torch.broadcast_to(self.get_masked_W(), (batch, self.n_x, self.n_x))
        # print(masked_W_broadcasted.type()) # debug
        # print(x.type())
        weighted_sum = torch.bmm(masked_W_broadcasted, x)

        # epsilon*phi(Sigma+u)-alpha*x    fixed envelope function
        # self.check_param() # debug
        return eps * torch.tanh(weighted_sum + u) - alpha * x

    def get_masked_W(self):
        # print((self.W * self.W_mask).shape) # debug
        return self.W * self.W_mask

    def check_param(self):
        w = torch.flatten(self.W.detach())
        w_min = torch.min(w)
        w_max = torch.max(w)

        eps = torch.max(F.softplus(self.epsilon_bs.detach()))
        alpha = torch.max(F.softplus(self.alpha_bs.detach()))

        print('---------------------------------------------------')
        print('Print param min max')
        print(f'W min : {w_min}\nW max : {w_max}\neps max : {eps}\nalpha : {alpha}')

    def check_param_grad(self):
        print('---------------------------------------------------')
        print('Print param grad min max')
        for key, val in self.params.items():
            max = torch.max(val.grad.detach())
            min = torch.min(val.grad.detach())

            print(f'{key} gradient \n  max : {max}\n  min : {min}')







class CellBox(nn.Module):
    def __init__(self, dt, nt, dxdt=None):
        super().__init__()

        self.dt = dt
        self.nt = nt
        self.nx = 0  # number of nodes

        self.ode_solver = None
        self._dxdt = None

        self.gradient_zero_from = 87 # temporary code

        self.config = None

    @staticmethod
    def build_model(args, config):
        # construct and build dxdt
        # this could have been done outside and added to the attribute but fixed like this for now

        cb = CellBox(args.dt, args.nt)

        cb._dxdt = Ode()

        # get node topology from config
        # print(config.keys()) # debug
        node_part_idx_ls = config["node_part_idxes"]
        node_subs_collection = get_node_ss_clt(node_part_idx_ls)
        if "node_adj_mat" in config:
            node_adj_mat = np.array(config["node_adj_mat"])
        else:
            # read from file
            node_adj_mat = pd.read_csv(config["node_adj_mat_path"], sep=',', header=None)
        cb._dxdt.build_model(node_part_idx_ls[-1], node_subs_collection, node_adj_mat) # think about how to pass weight info

        # set parameters of model
        cb.params = cb._dxdt.get_param_dict() # get them from dxdt

        # set ode_solver
        cb.ode_solver = get_ode_solver(args.ode_solver)

        # set configuration
        cb.config = config

        return cb

    def forward(self, y0, u):

        # drug input being sparse is not yet implemented
        # if isinstance(u, tf.SparseTensor):
        #     u_t = tf.sparse.to_dense(tf.sparse.transpose(u))
        # else:
        #     u_t = tf.transpose(u)

        batch_size, _, _ = y0.shape
        u_t = u.reshape(batch_size, 1, -1)
        # print(f'u shape: {u.shape}')
        # print(f'u_t: {u_t.shape}')

        # Solve the ODE
        # print(self.ode_solver) # debug
        ys = self.ode_solver.solve(x=y0,
                                   u_t=u_t,
                                   dt=self.dt,
                                   nt=self.nt,
                                   ode=self._dxdt,
                                   n_activity_nodes=self.gradient_zero_from)

        # [nt, n_x, batch_size]
        # ys = ys[-self.args.ode_last_steps:]
        # ys = ys[-5:] # temporary fix
        # [n_iter_tail, n_x, batch_size]

        # Take the last value of ys
        yhat = ys[-1].reshape(batch_size, -1, 1)

        # implement convergence
        # mean, sd = tf.nn.moments(ys, axes=0)
        #
        # dxdt = self._dxdt(ys[-1], u_t)
        # # [n_x, batch_size] for last ODE step
        # convergence_metric = tf.concat([mean, sd, dxdt], axis=0)

        convergence_metric = 0
        return yhat
        # return convergence_metric, yhat

class GeneralModel(nn.Module):
    def __init__(self, dt, nt, dxdt=None):
        super().__init__()

        self.dt = dt
        self.nt = nt
        self.nx = 0  # number of nodes

        self.ode_solver = None
        self._dxdt = None

        self.config = None

    @staticmethod
    def build_model(args, config):
        cb = CellBox(args.dt, args.nt)

        cb._dxdt = Ode() # map

        # get node topology from config
        # print(config.keys()) # debug
        node_part_idx_ls = config["node_part_idxes"]
        node_subs_collection = get_node_ss_clt(node_part_idx_ls)
        if "node_adj_mat" in config:
            node_adj_mat = np.array(config["node_adj_mat"])
        else:
            # read from file
            node_adj_mat = pd.read_csv(config["node_adj_mat_path"], sep=',', header=None)
        cb._dxdt.build_model(node_part_idx_ls[-1], node_subs_collection,
                             node_adj_mat)  # think about how to pass weight info

        # set parameters of model
        cb.params = cb._dxdt.get_param_dict()  # get them from dxdt

        # set ode_solver
        cb.ode_solver = get_ode_solver(args.ode_solver)

        # set configuration
        cb.config = config

        return cb

    def forward(self, y0, u):

        # drug input being sparse is not yet implemented
        # if isinstance(u, tf.SparseTensor):
        #     u_t = tf.sparse.to_dense(tf.sparse.transpose(u))
        # else:
        #     u_t = tf.transpose(u)

        batch_size, _, _ = y0.shape
        u_t = u.reshape(batch_size, 1, -1)
        # print(f'u shape: {u.shape}')
        # print(f'u_t: {u_t.shape}')

        # Solve the ODE
        # print(self.ode_solver) # debug
        ys = self.ode_solver.solve(x=y0,
                                   u_t=u_t,
                                   dt=self.dt,
                                   nt=self.nt,
                                   ode=self._dxdt,
                                   n_activity_nodes=self.gradient_zero_from)

        # [nt, n_x, batch_size]
        # ys = ys[-self.args.ode_last_steps:]
        # ys = ys[-5:] # temporary fix
        # [n_iter_tail, n_x, batch_size]

        # Take the last value of ys
        yhat = ys[-1].reshape(batch_size, -1, 1)

        # implement convergence
        # mean, sd = tf.nn.moments(ys, axes=0)
        #
        # dxdt = self._dxdt(ys[-1], u_t)
        # # [n_x, batch_size] for last ODE step
        # convergence_metric = tf.concat([mean, sd, dxdt], axis=0)

        convergence_metric = 0
        return yhat
        # return convergence_metric, yhat


# if __name__ == '__main__':
#     args  = 0
#     model = factory(args)
