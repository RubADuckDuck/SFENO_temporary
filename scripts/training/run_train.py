import numpy as np
import torch
import os
import matplotlib.pyplot as plt

from sfeno.models import StructInfoLayer
from sfeno.models import general_model
from sfeno.models import torchlightning_wrapper
from sfeno.trainers import BasicTrainer

from sfeno.datasets.dataset import get_dataloaders
from sfeno.datasets.data_converter import save_prediction_n_target

from sfeno.utils.create_adj_mat import read_sif
from sfeno.utils.model_settings import Args
from sfeno.utils.model_settings import Config

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import LightningDataModule

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-w", "--num_workers", type=int, dest="num_workers", action="store", default=0)  # extra value
parser.add_argument("-b", "--num_batch", type=int, dest="num_batch", action="store", default=1)
parser.add_argument("-g", "--num_gpu", type=int, dest="num_gpu", action="store", default=1)
parser.add_argument("--debug", action="store_true", dest="debug", default=False)
parser.add_argument("-e", "--epoch", type=int, dest="num_epoch", action="store", default=0)
parser.add_argument("--ddp", action="store_true", dest="ddp", default=False)
parser.add_argument("-d", "--data_path", type=str, action="store", default='sfeno/datasets/korkut_2015a/sfeno_data')
parser.add_argument("--nn_ode", action='store_true', default=False)
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

def save_tensor(mat, path):
    np_arr = mat.detach().cpu().numpy()
    np.save(path, np_arr)

if __name__ == '__main__':
    debug = pargs.debug
    debug = True # debug
    detect_anomaly = False
    do_staging = 2


    args = Args()

    data_dir = 'sfeno/datasets/synthesized'

    # get configuration
    if debug:
        config_path = "configs/bio_info_config_debug.json"
    else:
        config_path = "configs/bio_info_config.json"
    config = Config(config_path)

    config.config['data_path'] = pargs.data_path
    # config.config['data_path'] = 'sfeno/datasets/korkut_2015a/sfeno_data'
    # config.config['data_path'] = 'sfeno/datasets/molinelli_2013/sfeno_data'

    data_path = config.config['data_path']

    config.config['batchsize'] = config.config['batchsize'] if pargs.num_batch==0 else pargs.num_batch
    config.config['max_epoch'] = config.config['max_epoch'] if pargs.num_epoch==0 else pargs.num_epoch

    trainloader, validloader, testloader = get_dataloaders(config)



    stages = config.config['stages']

    # load data
    n_x = config.config['n_x']

    lit_datamodule = DataModule(trainloader, validloader, testloader)
    # create dxdt

    if pargs.nn_ode:
        lit_sfeno = general_model.LitSfeno_v2(config=config, args=args,
                                              ud_class_nm=general_model.NN_dxdt)
    else:
        lit_sfeno = general_model.LitSfeno_v2(config=config, args=args,
                                              ud_class_nm=general_model.Example_Cellbox)

    print([i.shape for i in lit_sfeno.parameters()])


    if pargs.ddp:
        trainer = Trainer(devices=pargs.num_gpu,
                          accelerator="gpu",
                          max_epochs=config.config['max_epoch'],
                          strategy="ddp_find_unused_parameters_false")
    else:
        trainer = Trainer(devices=pargs.num_gpu,
                          accelerator="gpu",
                          max_epochs=config.config['max_epoch'], )

    trainer.fit(lit_sfeno, datamodule=lit_datamodule)

    save_tensor(lit_sfeno.model.dxdt.net_model.get_network_weight(),
                os.path.join(config.config['data_path'], 'net_mat.npy'))

    pred_ls = trainer.predict(lit_sfeno, datamodule=lit_datamodule)

    pred = torch.flatten(pred_ls[0][0]).detach().numpy()
    ans = torch.flatten(pred_ls[0][1]).detach().numpy()

    save_prediction_n_target(pred, ans, data_path)









