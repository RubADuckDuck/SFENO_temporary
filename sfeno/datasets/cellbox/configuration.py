import os
from os.path import join as pjoin


class Configuration:
    def __init__(self,
                 fpath_pert=None,
                 fpath_expr=None,
                 fpath_node_idx=None):

        dpath_root = os.path.dirname(__file__)
        if not fpath_pert:
            fpath_pert = pjoin(dpath_root, "pert.csv")
            
        if not fpath_expr:
            fpath_expr = pjoin(dpath_root, "expr.csv")
        
        if not fpath_node_idx:
            fpath_node_idx = pjoin(dpath_root, "node_index.csv")
        
        self.fpath_pert = fpath_pert  
        self.fpath_expr = fpath_expr
        self.fpath_node_idx = fpath_node_idx

        self.pert = None
        self.expr = None
        self.node_idx = None
        self.nexp = None # number of nodes
        self.num_data = None # number of data(permutation

        self.train_ratio = 0.6
        self.valid_ratio = 0.8

        self.batch_size = 4