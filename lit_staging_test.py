import numpy as np
import torch

from sfeno.models import StructInfoLayer
from sfeno.models import general_model
from sfeno.models import torchlightning_wrapper
from sfeno.trainers import BasicTrainer


from sfeno.datasets.synthesized import get_dataloaders


from sfeno.utils.create_adj_mat import read_sif
from sfeno.utils.model_settings import Args
from sfeno.utils.model_settings import Config

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import LightningDataModule

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-w", "--num_workers", type=int, dest="num_workers", action="store", default=0)          # extra value
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
    debug = True
    detect_anomaly = False
    do_staging = 2
    training_implemented = True
    version = 2

    args = Args()

    # get configuration
    if debug:
        config_path = "configs/rp_syn_lit_config_debug.json"
    else:
        config_path = "configs/rp_syn_lit_config.json"
    config = Config(config_path)

    print(config)
    print(config)

    stages = config.config['stages']

    trainloader, validloader, testloader = get_dataloaders(config, num_workers=pargs.num_workers)
    lit_datamodule = DataModule(trainloader, validloader, testloader)
    # create dxdt

    if version==1:
        dxdt = general_model.ExampleUserImplemented(config=config)

        model = general_model.GeneralModel.build_model(config, args, dxdt=dxdt)

        lit_sfeno = torchlightning_wrapper.LitSfeno(model)

    elif version==2:
        lit_sfeno = general_model.LitSfeno_v2(config=config, args=args,
                                              ud_class_nm=general_model.ExampleUserImplemented)

    trainer = Trainer(devices=1, accelerator="gpu", max_epochs=200)

    trainer.fit(lit_sfeno, datamodule=lit_datamodule)






