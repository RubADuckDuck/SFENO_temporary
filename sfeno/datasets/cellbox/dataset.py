import os
from os.path import join as pjoin

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sfeno.datasets.cellbox.configuration import Configuration


class PertExpDataset(Dataset):
    def __init__(self, pert: pd.DataFrame, exp: pd.DataFrame):
        self.pert_ds = torch.tensor(pert.values)
        self.expr_ds = torch.tensor(exp.values)
        self.pert_form = 'fix x'  # 'by u' or 'fix x' fix level of node by input perturbation u

    def __len__(self):
        return len(self.pert_ds)

    def __getitem__(self, idx):
        cur_pert = self.pert_ds[idx]
        cur_expr = self.expr_ds[idx]

        cur_pert = cur_pert.reshape(1, 1, -1)
        cur_expr = cur_expr.reshape(1, 1, -1)

        # print(cur_pert) # debug
        if self.pert_form == 'by u':
            y0 = torch.full_like(cur_expr, 0)
        elif self.pert_form == 'fix x':
            y0 = cur_pert

        # print(y0.shape, cur_pert.shape, cur_expr.shape) # debug

        return (y0, {'mu': cur_pert}), cur_expr # where can i find x0??



def get_dataloaders(cfg: Configuration):
    ''' random partition '''
    df_pert = pd.read_csv(cfg.config['pert_file'], header=None, dtype=np.float32)
    df_expr = pd.read_csv(cfg.config['expr_file'], header=None, dtype=np.float32)
    df_node_idx = pd.read_csv(cfg.config['node_index_file'], header=None)

    print(f'Pert Shape : {df_pert.shape}')
    print(f'Expr Shape : {df_expr.shape}')
    print(f'Node Index Shape : {df_node_idx.shape}')

    train_ratio = cfg.config['trainset_ratio']
    valid_ratio = cfg.config['validset_ratio']

    cfg.config['num_data'], cfg.config['n_x'] = df_pert.shape
    print(f'Number of Nodes: {cfg.config["n_x"]}')
    print(f'Number of Permutations: {cfg.config["num_data"]}')

    nexp = cfg.config['n_x']
    num_data = cfg.config['num_data']
    ntrain = int(num_data * train_ratio)
    nvalid = int(num_data * valid_ratio)

    # Randomly partitioned datasets
    
    # [OPTION #1] CellBox implementation
    try:
        raise
        # random_pos = np.genfromtxt('random_pos.csv', defaultfmt='%d')
    except Exception:
      random_pos = np.random.choice(range(num_data), num_data, replace=False)
      np.savetxt('random_pos.csv', random_pos, fmt='%d')


    # [OPTION #2]
    # if cfg.fpath_random_pos:
    #     pass
    #     # Read file....
    #     # random_pos = ...
    # 
    # else:
    #     random_pos = np.random.choice(range(num_data), num_data, replace=False)

    # [OPTION #3]
    # random_pos = np.arange(num_data)
    # np.random.shuffle(random_pos)

    pos = {}

    pos['train'] = random_pos[:ntrain]
    pos['valid'] = random_pos[ntrain:nvalid]
    pos['test'] = random_pos[nvalid:]

    print(pos) # debug

    pert_train = df_pert.iloc[pos['train']]
    pert_valid = df_pert.iloc[pos['valid']]
    pert_test = df_pert.iloc[pos['test']]
    
    expr_train = df_expr.iloc[pos['train']]
    expr_valid = df_expr.iloc[pos['valid']]
    expr_test = df_expr.iloc[pos['test']]

    # debug
    # print(pert_train.shape)

    train_ds = PertExpDataset(pert_train, expr_train, )
    valid_ds = PertExpDataset(pert_valid, expr_valid, )
    test_ds = PertExpDataset(pert_test, expr_test, )

    train_dataloader = DataLoader(train_ds, batch_size=cfg.config['batchsize'])
    valid_dataloader = DataLoader(valid_ds, batch_size=cfg.config['batchsize'])
    test_dataloader = DataLoader(test_ds, batch_size=cfg.config['batchsize'])

    return train_dataloader, valid_dataloader, test_dataloader






