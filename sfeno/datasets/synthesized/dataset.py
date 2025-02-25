import os
from os.path import join as pjoin

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sfeno.datasets.synthesized import synthetic_data_generator

import os
import random


class PertExpDataset(Dataset):
    def __init__(self, pert: pd.DataFrame, exp: pd.DataFrame):
        super().__init__()
        self.pert_ds = torch.tensor(pert.values, dtype=torch.float32, requires_grad=False)
        self.expr_ds = torch.tensor(exp.values, dtype=torch.float32, requires_grad=False)
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

        return (y0.detach(), {'mu': cur_pert.detach()}), cur_expr.detach() # where can i find x0??




def create_and_save_data(path, n_x):
    ex_pert_data = synthetic_data_generator.RandomExpressionPerturbationGenerator(n_x)

    print(f'Generating data of network size {n_x}...')
    pert, expr = ex_pert_data.create_datasets()

    save_dir = save_array_as_csv(path, n_x, pert, "pert.csv")
    save_array_as_csv(save_dir, n_x, expr, "expr.csv", new=False)

def check_n_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_array_as_csv(path, n_x, arr, name, new = True):
    if new:
        idx_limit = 100

        dir_idx = 0

        # check if n_x exist and if not mkdir
        n_x_dir = os.path.join(path, str(n_x))

        # mkdir
        check_n_mkdir(n_x_dir)


        while True:
            cur_dir_idx = os.path.join(n_x_dir, str(dir_idx))
            if os.path.exists(cur_dir_idx):
                dir_idx += 1
                if dir_idx == idx_limit:
                    raise IndexError
                else:
                    continue
            else:
                os.makedirs(cur_dir_idx)
                break

        result_path = os.path.join(cur_dir_idx, name)

        np.savetxt(result_path, arr, delimiter=",")

        return cur_dir_idx
    else:
        result_path = os.path.join(path, name)

        np.savetxt(result_path, arr, delimiter=",")

        return path

def get_n_x_network_df(path, n_x, idx=None):
    n_x_dir = os.path.join(path, str(n_x))

    only_dir = [d for d in os.listdir(n_x_dir)]
    print(only_dir)

    if idx is None:
        rand_idx = random.randrange(0, len(only_dir), 1)

        idx_dir = only_dir[rand_idx]
    else:
        idx_dir = str(idx)

    data_path = os.path.join(n_x_dir, idx_dir)

    print(f"Reading data from {data_path}...")

    pert_name = 'pert.csv'
    expr_name = 'expr.csv'
    pert = np.loadtxt(os.path.join(n_x_dir, idx_dir, pert_name), delimiter=',')
    expr = np.loadtxt(os.path.join(n_x_dir, idx_dir, expr_name), delimiter=',')

    # print(pert)
    # print(expr)

    pert_df = pd.DataFrame(pert)
    expr_df = pd.DataFrame(expr)

    return pert_df, expr_df, n_x, idx_dir, data_path


def get_dataloaders(cfg, pert_df, expr_df, num_workers=0):
    ''' random partition '''


    n_x = cfg.config['n_x']

    df_pert = pert_df
    df_expr = expr_df

    print(f'Pert Shape : {df_pert.shape}')
    print(f'Expr Shape : {df_expr.shape}')

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

    # print(pos)  # debug

    pert_train = df_pert.iloc[pos['train']]
    pert_valid = df_pert.iloc[pos['valid']]
    pert_test = df_pert.iloc[pos['test']]

    expr_train = df_expr.iloc[pos['train']]
    expr_valid = df_expr.iloc[pos['valid']]
    expr_test = df_expr.iloc[pos['test']]

    # debug
    # print(pert_train.shape)

    train_ds = PertExpDataset(pert_train, expr_train)
    valid_ds = PertExpDataset(pert_valid, expr_valid)
    test_ds = PertExpDataset(pert_test, expr_test)

    n_w = num_workers

    train_dataloader = DataLoader(train_ds, batch_size=cfg.config['batchsize'], num_workers=n_w, pin_memory=True)
    valid_dataloader = DataLoader(valid_ds, batch_size=cfg.config['batchsize'], num_workers=n_w, pin_memory=True)
    test_dataloader = DataLoader(test_ds, batch_size=cfg.config['batchsize'], num_workers=n_w, pin_memory=True)

    return train_dataloader, valid_dataloader, test_dataloader
