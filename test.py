import torch

from sfeno.models import CellBox
from sfeno.trainers import BasicTrainer

from sfeno.datasets.cellbox import get_dataloaders
from sfeno.datasets.cellbox import Configuration

import json

# to implement staging

stages=[
    {
        'nT': 100,
        'sub_stages': [
            {'lr_val': 0.1, 'l1lambda': 0.01, 'n_iter_patience': 1000},
            {'lr_val': 0.01, 'l1lambda': 0.01},
            {'lr_val': 0.01, 'l1lambda': 0.0001},
            {'lr_val': 0.001, 'l1lambda': 1e-05}
        ]
     },
    {
        'nT': 200,
        'sub_stages': [
            {'lr_val': 0.001, 'l1lambda': 0.0001}
        ]
    },
    {
        'nT': 400,
        'sub_stages': [
            {'lr_val': 0.001, 'l1lambda': 0.0001}
        ]
    }
]



class Args:
    def __init__(self):
        self.dt = torch.tensor(0.1,requires_grad=False, dtype=torch.float32)
        self.nt = 400
        self.ode_solver = "heun" # "torchdiffeq"


if __name__ == '__main__':
    debug = True
    detect_anomaly = False
    do_staging = 2
    training_implemented = True

    args = Args()

    # get configuration
    if debug:
        config_path = "rp_debug.json"
    else:
        config_path = "rp_config.json"
    with open(config_path, "r") as st_json:
        config = json.load(st_json)

    print(config)
    print(config)

    model = CellBox.build_model(args, config)
    trainloader, validloader, testloader = get_dataloaders(config)
    
    trainer = BasicTrainer(model,
                           trainloader=trainloader,
                           validloader=validloader,
                           config = config)
    # trainer.train(200)

    if detect_anomaly:
        with torch.autograd.detect_anomaly():
            if do_staging==1:
                for st_idx, cur_stg in enumerate(stages):
                    cur_sub_stges = cur_stg['sub_stages']
                    cur_nt = cur_stg['nT']

                    print(f"Stage {st_idx} ==============================")
                    print(f'  nT: {cur_nt}')
                    for sst_idx, cur_sstg in enumerate(cur_sub_stges):
                        print(f'    Substage: {cur_sstg}')

                        if training_implemented:
                            trainer.train_substage(cur_sstg) # need fixing arguments dont match

            elif do_staging==2:
                for st_idx, cur_stg in enumerate(stages):
                    cur_sub_stges = cur_stg['sub_stages']
                    cur_nt = cur_stg['nT']

                    print(f"Stage {st_idx} ==============================")
                    print(f'  nT: {cur_nt}')

                    # cellbox
                    model.nt = cur_nt

                    if training_implemented:
                        trainer.train_by_ss(cur_stg)

            else:
                trainer.train(200)
    else:
        if do_staging == 1:
            for st_idx, cur_stg in enumerate(stages):
                cur_sub_stges = cur_stg['sub_stages']
                cur_nt = cur_stg['nT']

                print(f"Stage {st_idx} ==============================")
                print(f'  nT: {cur_nt}')
                for sst_idx, cur_sstg in enumerate(cur_sub_stges):
                    print(f'    Substage: {cur_sstg}')

                    if training_implemented:
                        trainer.train_substage(cur_sstg)  # need fixing arguments dont match

        elif do_staging == 2:
            for st_idx, cur_stg in enumerate(stages):
                cur_sub_stges = cur_stg['sub_stages']
                cur_nt = cur_stg['nT']

                print(f"Stage {st_idx} ==============================")
                print(f'  nT: {cur_nt}')

                # cellbox
                model.nt = cur_nt

                if training_implemented:
                    trainer.train_by_ss(cur_stg)

        else:
            trainer.train(200)