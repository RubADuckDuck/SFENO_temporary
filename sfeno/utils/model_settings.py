import torch
import json
import os
from sfeno.utils.create_adj_mat import read_sif, get_adj_index_given

class Args:
    def __init__(self):
        self.dt = torch.tensor(0.1,requires_grad=False, dtype=torch.float32)
        self.nt = 400
        self.ode_solver = "heun" # "torchdiffeq"

class Config:
    def __init__(self, path):
        with open(path, "r") as st_json:
            self.config = json.load(st_json)

            try:
                self.adj_mat, self.n2i, self.node_nm_ls = read_sif(self.config["network_struct_file"])
            except:
                self.adj_mat = 1
                self.n2i = None
                self.node_nm_ls = None

        self.node_name_ls_path = ''
        self.network_file_path = ''

        self.net_file_nm = 'network.sif'
        self.node_file_nm = 'node_Index.json'

    def load_paths_n_info(self):
        data_path = self.config['data_path']
        self.node_name_ls_path = os.path.join(data_path, self.node_file_nm)
        self.network_file_path = os.path.join(data_path,'..',self.net_file_nm)

        # load node name list
        with open(self.node_name_ls_path, "r") as temp_json:
            self.node_nm_ls = json.load(temp_json)
            print(self.node_nm_ls)
            self.config["n_x"] = len(self.node_nm_ls)

        # load adj mat
        self.adj = get_adj_index_given(self.network_file_path, self.node_nm_ls)
        print(self.adj_mat)

    def read_adj_mat(self, path):
        self.adj_mat = get_adj_index_given()