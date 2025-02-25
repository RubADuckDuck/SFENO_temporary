import numpy as np
import pytorch_lightning
import torch

from sfeno.models import StructInfoLayer
from sfeno.models import general_model
from sfeno.models import torchlightning_wrapper
from sfeno.trainers import BasicTrainer

from sfeno.datasets.synthesized import get_dataloaders
from sfeno.datasets.synthesized import synthetic_data_generator
from sfeno.datasets.synthesized import dataset

from sfeno.utils.create_adj_mat import read_sif
from sfeno.utils.model_settings import Args
from sfeno.utils.model_settings import Config

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import LightningDataModule

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-w", "--num_workers", type=int, dest="num_workers", action="store", default=0)  # extra value
parser.add_argument("-b", "--num_batch", type=int, dest="num_batch", action="store", default=0)
parser.add_argument("-g", "--num_gpu", type=int, dest="num_gpu", action="store", default=1)
parser.add_argument("--debug", action="store_true", dest="debug", default=False)
parser.add_argument("-e", "--epoch", type=int, dest="num_epoch", action="store", default=0)
parser.add_argument("--ddp", action="store_true", dest="ddp", default=False)
parser.add_argument("--network_size", type=int, dest="network_size", action="store", default=0)
parser.add_argument("--data_idx", type=int, action="store", default=0)
parser.add_argument("-c", "--checkpoint_path", type=str, default=None, action="store")
pargs = parser.parse_args()


class DataModule(LightningDataModule):
    def __init__(self, tn, vl, ts):
        super().__init__()
        self.training_dl = tn
        self.validation_dl = vl
        self. test_dl = ts

    def train_dataloader(self):
        return self.training_dl

    def val_dataloader(self):
        return self.validation_dl

    def test_dataloader(self):
        return self.test_dl

    def predict_dataloader(self):
        return self.test_dl

if __name__ == '__main__':
    do_staging = 2
    version = 2

    args = Args()

    data_dir = 'sfeno/datasets/synthesized'


    cp = pargs.checkpoint_path
    if cp is None:
        cp = 'lightning_logs/version_23/checkpoints/epoch=0-step=4.ckpt'

    # get configuration

    config_path = "configs/rp_syn_lit_config.json"

    config = Config(config_path)

    config.config['data_path'] = 'sfeno/datasets/synthesized'

    config.config['batchsize'] = config.config['batchsize'] if pargs.num_batch==0 else pargs.num_batch
    config.config['max_epoch'] = config.config['max_epoch'] if pargs.num_epoch==0 else pargs.num_epoch
    config.config['n_x'] = config.config['n_x'] if pargs.network_size == 0 else pargs.network_size

    stages = config.config['stages']

    # load data
    n_x = config.config['n_x']
    pert_df, expr_df, _, data_idx = dataset.get_n_x_network_df(data_dir, n_x, idx=pargs.data_idx)

    # print(pert_df)
    # print(expr_df)

    trainloader, validloader, testloader = get_dataloaders(config, pert_df, expr_df, num_workers=pargs.num_workers)

    lit_datamodule = DataModule(trainloader, validloader, testloader)
    # create dxdt

    # load checkpoint model
    lit_sfeno = general_model.LitSfeno_v2(config=config, args=args,
                                          ud_class_nm=general_model.Example_Cellbox)
    lit_sfeno.load_from_checkpoint(config=config, args=args, ud_class_nm=general_model.Example_Cellbox, checkpoint_path=cp)

    trainer = Trainer(devices=pargs.num_gpu, accelerator="gpu", strategy="ddp_find_unused_parameters_false")

    trainer.test(lit_sfeno, datamodule=lit_datamodule, verbose=True)