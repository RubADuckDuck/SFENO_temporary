import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

def create_mask(n, node_subs_collection, adj_mat):
    W_mask = torch.zeros((n,n))

    n_subset = len(node_subs_collection)

    for i_in in range(n_subset):
        for j_out in range(n_subset):
            connection = adj_mat[i_in,j_out]

            for i in node_subs_collection[i_in]:
                for j in node_subs_collection[j_out]:
                    W_mask[i,j] = connection

    return W_mask

class F_dxdt(nn.Module):
    def __init__(self):
        super().__init__()

        self.epsilon = 0
        self.W = 0
        self.alpha = 0

        # u, strength of perturbation on node is currently fixed as a constant
        # meaning only shows the case where the drug effect is assumed to be constant
        self.u = 0
        # if possible make u as a function when given t returns the strength of perturbation

    def build_model(self,n_x, i_pt_protein_nodes, i_pt_drg_activity_nodes):
        self.init_params(n_x, i_pt_protein_nodes, i_pt_drg_activity_nodes)
        return

    def init_params(self, n_x, i_pt_protein_nodes, i_pt_drg_activity_nodes):
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
        W = torch.Tensor(np.random.normal(0.01, size=(n_x, n_x), requires_grad=True))

        W_mask = (1.0 - np.diag(np.ones([n_x])))
        W_mask[i_pt_drg_activity_nodes:, :] = np.zeros([n_x - i_pt_drg_activity_nodes, n_x])
        W_mask[:, i_pt_protein_nodes:i_pt_drg_activity_nodes] = np.zeros([n_x, i_pt_drg_activity_nodes - i_pt_protein_nodes])
        W_mask[i_pt_protein_nodes:i_pt_drg_activity_nodes, i_pt_drg_activity_nodes:] = np.zeros([i_pt_drg_activity_nodes - i_pt_protein_nodes,
                                                                                n_x - i_pt_drg_activity_nodes])
        W_mask = torch.Tensor(W_mask, requires_grad=False)

        self.W = W
        self.W_mask = W_mask

        eps = torch.tensor(np.ones((n_x, 1)), requires_grad=True)
        alpha = torch.tensor(np.ones((n_x, 1)), requires_grad=True)
        self.alpha = F.softplus(alpha)
        self.epsilon = F.softplus(eps)

        return

    def set_mu(self, u):
        '''
        :param u:
        :return:

            called to set u differently for each perterbation(data sample)
        '''
        self.u = u

        return


    def get_params(self):
        return self.epsilon, self.W, self.alpha

    def forward(self, t, x):
        """
        calculate the derivatives dx/dt in the ODEs
            something to note is that second argument is not time,
            but it quantifies the strength of the perturbation on target
        """

        weighted_sum = torch.matmul(self.get_masked_W(), x)

        # epsilon*phi(Sigma+u)-alpha*x
        return self.epsilon * torch.tanh(weighted_sum + self.u) - self.alpha * x

    def get_masked_W(self):
        return self.W * self.W_mask


class F_dxdt_gnr(nn.Module):
    def __init__(self):
        super().__init__()

        self.epsilon = 0
        self.W = 0
        self.alpha = 0

        # u, strength of perturbation on node is currently fixed as a constant
        # meaning only shows the case where the drug effect is assumed to be constant
        self.u = 0
        # if possible make u as a function when given t returns the strength of perturbation

    def build_model(self,n_x, node_subs_collection, adj_mat):
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
        W = torch.Tensor(np.random.normal(0.01, size=(n_x, n_x), requires_grad=True))

        W_mask = create_mask(n_x, node_subs_collection, adj_mat)
        W_mask = torch.Tensor(W_mask, requires_grad=False)

        self.W = W
        self.W_mask = W_mask

        eps = torch.tensor(np.ones((n_x, 1)), requires_grad=True)
        alpha = torch.tensor(np.ones((n_x, 1)), requires_grad=True)
        self.alpha = F.softplus(alpha)
        self.epsilon = F.softplus(eps)

        return

    def set_mu(self, u):
        '''
        :param u:
        :return:

            called to set u differently for each perterbation(data sample)
        '''
        self.u = u

        return


    def get_params(self):
        return self.epsilon, self.W, self.alpha

    def forward(self, t, x):
        """
        calculate the derivatives dx/dt in the ODEs
            something to note is that second argument is not time,
            but it quantifies the strength of the perturbation on target
        """

        weighted_sum = torch.matmul(self.get_masked_W(), x)

        # epsilon*phi(Sigma+u)-alpha*x
        return self.epsilon * torch.tanh(weighted_sum + self.u) - self.alpha * x

    def get_masked_W(self):
        return self.W * self.W_mask


def create_mask(n, node_subs_collection, adj_mat):
    W_mask = torch.zeros((n,n))

    n_subset = len(node_subs_collection)

    for i_in in range(n_subset):
        for j_out in range(n_subset):
            connection = adj_mat[i_in,j_out]

            for i in node_subs_collection[i_in]:
                for j in node_subs_collection[j_out]:
                    W_mask[i,j] = connection

    return W_mask

class OdeEquationGenerator:
    def __init__(self):
        self.instruction = None
    
    def build_generator(self, instruction):
        pass   

    def generate_ode_eq_instance(self):
        return Ode_module

class GeneralOde:
    def __init__(self):
        self.

class OperationInst:
    def __init__(
        self,
        opr_instance_data,
        p_instruction,
        prcd_idx2opr_inst_idx,
        input_idx2nm
    ):
        self.nm2arg = dict()

        self.opr_inst_data = opr_instance_data
        self.p_instruction = p_instruction
        self.prcd_idx2opr_inst_idx = prcd_idx2opr_inst_idx
        self.input_idx2nm = input_idx2nm

    def forward(self, *args):
        for idx, arg in enumerate(args):
            # with argument index get name of argument
            cur_nm = self.input_idx2nm[idx]
            # with name add instance add name, arg_inst pair to dictionary
            self.nm2arg[cur_nm] = arg

        for prcd_idx, procedure in enumerate(self.p_instruction):
            # get operator name, argument name, and results argument name
            opr_nm, arg_nm_ls, new_arg_nm = procedure
            # get operation index for operation data from prcd_idx2opr_inst_idx
            opr_idx = self.prcd_idx2opr_inst_idx[prcd_idx]

            # calculate result
            cur_res = self.solve_by_ref(opr_nm, opr_idx, arg_nm_ls)

            # add arg_name and arg_instance pair to dictionary
            self.nm2arg[new_arg_nm] = cur_res
        
        # return last calculation
        return cur_res

    def solve_by_ref(self, opr_nm, opr_idx, arg_nm_ls):
        # get argument objects
        arg_obj_ls = [self.nm2arg[nm] for nm in arg_nm_ls]
        # get operator objects
        opr_inst = self.prcd_idx2opr_inst_idx[opr_nm][opr_idx]
        # forward
        return opr_inst(*arg_nm_ls)