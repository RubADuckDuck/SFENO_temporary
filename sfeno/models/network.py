import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

# class NetworkLayer(nn.Module):
#     def __init__(self, adj_mat, node_nmes=None):
#         super().__init__()
#
#         self.adj_mat = adj_mat
#         self.node_names = node_nmes
#         self.node_num = len(self.adj_mat)
#
#         self.w_frame = torch.zeros((self.node_num, self.node_num), requires_grad=False)
#         self.params = dict()
#
#         cur_adj_val = 0
#
#             # 0
#         def zero_param(x):
#             return x * 0
#
#             # R
#         def R_param(x):
#             return x
#
#             # R+
#         def R_pos_param(x):
#             return F.softplus(x)
#
#             # R-
#         def R_ngt_param(x):
#             return(-1) * F.softplus(x)
#
#         self.adj_val2get_param_func = {
#             0: zero_param,
#             1: R_param,
#             2: R_pos_param,
#             3: R_ngt_param
#         }
#
#
#
#     def init_params(self):
#         if self.node_names is not None:
#             # create mask
#             for i, eff_nn in enumerate(self.node_names):
#                 for j, rcv_nn in enumerate(self.node_names):
#                     self.params[(i,j)] = nn.parameter.Parameter(data=torch.zeros(1), requires_grad=True)
#                     self.register_parameter(f'{i} {j}', self.params[(i, j)])
#         else:
#             # create mask
#             for i in range(self.node_num):
#                 for j in range(self.node_num):
#                     self.params[(i,j)] = nn.parameter.Parameter(data=torch.zeros(1), requires_grad=True)
#                     self.register_parameter(f'{i} {j}', self.params[(i, j)])
#
#     def reload_mask(self):
#         self.w_frame = torch.zeros((self.node_num, self.node_num), requires_grad=False)
#         if self.node_names is not None:
#             for i, eff_nn in enumerate(self.node_names):
#                 for j, rcv_nn in enumerate(self.node_names):
#                     cur_adj_val = self.adj_mat[i, j]
#                     self.w_frame[i,j] = self.w_frame[i,j] + self.adj_val2get_param_func[cur_adj_val](self.params[(i,j)])
#         else:
#             for i in range(self.node_num):
#                 for j in range(self.node_num):
#                     cur_adj_val = self.adj_mat[i, j]
#                     self.w_frame[i,j] = self.w_frame[i,j] + self.adj_val2get_param_func[cur_adj_val](self.params[(i,j)])
#     def forward(self, x):
#         self.reload_mask() # test this is wastefull
#         w = self.w_frame
#
#         return torch.matmul(x, w)


class StructInfoLayer(nn.Module):
    def __init__(self, config, master):
        super().__init__()
        self.master = [master]

        if config.adj_mat==1:
            self.adj_mat = np.ones((config.config["n_x"],config.config["n_x"]))
        else:
            self.adj_mat = config.adj_mat

        self.node_names = config.node_nm_ls
        self.node_num = len(self.adj_mat)


        self.register_parameter(
            'W', nn.parameter.Parameter(torch.zeros((self.node_num, self.node_num), requires_grad=True))
        )
        # self.params = nn.ParameterDict({
        #         'W': nn.Parameter(torch.zeros((self.node_num, self.node_num), requires_grad=True))
        # })

        self.mask = torch.zeros(
            (self.node_num, self.node_num),
            requires_grad=False).type_as(next(self.parameters()))
        # self.register_buffer(name='mask',
        #                      tensor=self.zeros((self.node_num, self.node_num), requires_grad=False))
            # 0
        def zero_param(x):
            return x * 0

            # R
        def R_param(x):
            return x

            # R+
        def R_pos_param(x):
            return F.softplus(x)

            # R-
        def R_ngt_param(x):
            return(-1) * F.softplus(x)

        self.adj_val2get_param_func = {
            0: zero_param,
            1: R_param,
            2: R_pos_param,
            3: R_ngt_param
        }


    def init_maskes(self):
        self.adj_val2mask = dict()

        for adj_val, func in self.adj_val2get_param_func.items():
            self.adj_val2mask[adj_val] = torch.tensor(
                self.adj_mat == adj_val,
                requires_grad=False).type_as(next(self.parameters()))

    def init_params(self):
        self.init_maskes()

    def load_mask(self):
        self.mask = torch.zeros(
            (self.node_num, self.node_num),
            requires_grad=False).type_as(next(self.parameters()))

        for adj_val, func in self.adj_val2get_param_func.items():
            # print(self.params['W'].device) # debug
            # print(self.adj_val2mask[adj_val].device)
            mask = func(self.W * self.adj_val2mask[adj_val].type_as(next(self.parameters())))

            self.mask = self.mask + mask

    def mask_to(self):
        self.mask.type_as(next(self.parameters()))

    def forward(self, x):
        self.load_mask() # test this is wastefull if possible run this as callback on step function
        # w = self.mask.to(self.get_device())
        w = self.mask  #.type_as(next(self.parameters()))

        return torch.matmul(x, w)

    def get_network_weight(self):
        self.load_mask()
        return self.mask

    def call_master(self):
        return self.master[0]

    def get_device(self):
        # print(self.call_master().get_device(),'!!!!')
        # return self.call_master().get_device()
        return next(self.parameters()).device

