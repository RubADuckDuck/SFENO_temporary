import os.path

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sfeno.solvers import get_ode_solver
from sfeno.models import StructInfoLayer

import math
import pandas as pd

import pytorch_lightning as pl

class GeneralODE(nn.Module):
    def __init__(self, config, master):
        super().__init__()
        self.master = [master]

        self.params = nn.ParameterDict()
        self.params_max_min = dict()

        # self.general_model = None

        self.config = config
        self.stages = config.config['stages'] if 'stages' in config.config else False
        if self.stages:
            self.stage_bds = self.config.config['stage_boundary']

    def call_master(self):
        return self.master[0]

    def get_device(self):
        return self.call_master().get_device()

    def forward(self, t0, x, **kwargs):
        # not implemetned
        return 

    def check_params(self):
        if not self.params:
            print('no parameters in member variable params')
        else: 
            print('printing parameters min max')
            for nm, param in self.params.items():
                cur_max = torch.max(torch.flatten(self.param.detach()))
                cur_min = torch.min(torch.flatten(self.param.detach()))

                print(f'  ({nm})  min: {cur_min}, max: {cur_max}')

    def check_param_grads(self):
        if not self.params:
            print('no parameters in member variable params')
        else: 
            print('printing parameter gradients min max')
            for nm, param in self.params.items():
                cur_max = torch.max(torch.flatten(self.param.grad.detach()))
                cur_min = torch.min(torch.flatten(self.param.grad.detach()))

                print(f'  ({nm}_grad)  min: {cur_min}, max: {cur_max}')

    def epoch2stage_idx(self, epoch):
        change_hype = False
        for i, stage_bd in enumerate(self.stage_bds):
            if stage_bd == epoch:
                change_hype = True
                return i, change_hype
            elif i == len(self.stage_bds)-1:
                return i, change_hype
            elif stage_bd <= epoch and epoch < self.stage_bds[i+1]:
                return i, change_hype
            else:
                pass

class Example_Cellbox(GeneralODE):
    def __init__(self, config, master):
        super().__init__(config, master)

        self.net_model = StructInfoLayer(config, master=self)
        self.net_model.init_params()
        self.net_model.load_mask()

        self.register_parameter(
            'epsilon',nn.parameter.Parameter(F.softplus(torch.ones((config.config["n_x"]),
                                                                   dtype=torch.float32, requires_grad=True)))
        )

        self.register_parameter(
            'alpha', nn.parameter.Parameter(F.softplus(torch.ones((config.config["n_x"]),
                                                                    dtype=torch.float32, requires_grad=True)))
        )

        self.hyper_params = {
            'l1': 1e-4,
            'l2': 1e-4
        }

        self.mse = torch.nn.MSELoss()

        self.stages = config.config['stages']

    def forward(self, t0, x, arg_dict):
        ''' recieve permutation with kwargs'''
        y = self.net_model(x) + arg_dict['mu']
        y = torch.tanh(y)
        y = self.epsilon * y
        y = y - self.alpha * x
        return y

    # let user define loss function
    def loss_fn(self, output, target):
        l1 = torch.sum(torch.abs(self.net_model.W))
        l2 = torch.sum(torch.pow(self.net_model.W, 2))

        mse_loss = self.mse(torch.squeeze(output), torch.squeeze(target))

        total_loss = self.hyper_params['l1'] * l1 + self.hyper_params['l1'] * l2 + mse_loss
        return total_loss

    def on_after_backward(self):
        self.net_model.load_mask()
        return

    def training_epoch_end(self, outputs):
        cur_epoch = self.master[0].master[0].current_epoch
        if not self.stages:
            # no staging
            return
        else:
            idx_stage, change_hype = self.epoch2stage_idx(cur_epoch)
            stage = self.stages[idx_stage]
            if change_hype:
                lr = stage['lr_val'] if 'lr_val' in stage else self.config.config['default']['lr_val']
                l1 = stage['l1'] if 'l1' in stage else self.config.config['default']['l1']
                l2 = stage['l2'] if 'l2' in stage else self.config.config['default']['l2']
                nt = stage['nT'] if 'nT' in stage else self.config.config['default']['nT']
                print(' ')
                print('Changing Hyperparameters')
                print(f'lr {lr}, l1 {l1}, l2 {l2}, nt {nt}')
                self.master[0].master[0].lr = lr
                self.hyper_params['l1'] = l1
                self.hyper_params['l2'] = l2
                self.master[0].nt = nt

                self.master[0].master[0].configure_optimizers()

                self.save_net_weight(cur_epoch)
            else:
                pass

    def save_net_weight(self, epoch):
        d_path = self.config.config['data_path']

        name = f'net_mat_epoch{epoch}.npy'

        save_path = os.path.join(d_path, name)

        print(f'saveing network to {save_path}')

        save_tensor(self.net_model.get_network_weight(), save_path)

def save_tensor(mat, path):
    np_arr = mat.detach().cpu().numpy()
    np.save(path, np_arr)


class NN_dxdt(GeneralODE):
    def __init__(self, config, master):
        super().__init__(config, master)

        self.model = nn.Sequential(
            nn.Linear(config.config['n_x']*2, config.config['n_x']),
            nn.ReLU(),
            nn.Linear(config.config['n_x'], config.config['n_x']),
            nn.ReLU(),
            nn.Linear(config.config['n_x'], config.config['n_x']),
        )

        self.mse = torch.nn.MSELoss()

        self.stages = config.config['stages']

    def forward(self, t0, x, arg_dict):
        ''' recieve permutation with kwargs'''
        temp = torch.cat((x, arg_dict['mu']),dim=3)
        result = self.model.forward(temp)
        return result

    # let user define loss function
    def loss_fn(self, output, target):
        mse_loss = self.mse(torch.squeeze(output), torch.squeeze(target))
        # print(mse_loss)

        total_loss = mse_loss
        return total_loss

    def training_epoch_end(self, outputs):
        cur_epoch = self.master[0].master[0].current_epoch

        if not self.stages:
            # no staging
            return
        else:
            idx_stage, change_hype = self.epoch2stage_idx(cur_epoch)

            stage = self.stages[idx_stage]
            if change_hype:
                lr = stage['lr_val'] if 'lr_val' in stage else self.config.config['default']['lr_val']
                nt = stage['nT'] if 'nT' in stage else self.config.config['default']['nT']
                print(' ')
                print('Changing Hyperparameters')
                print(f'lr {lr}, nt {nt}')
                self.master[0].master[0].lr = lr
                self.master[0].nt = nt

                self.master[0].master[0].configure_optimizers()
            else:
                pass

class ExampleUserImplemented(GeneralODE):
        def __init__(self, config, master):
            super().__init__(config, master)

            self.net_model = StructInfoLayer(config, master=self)
            self.net_model.init_params()
            self.net_model.load_mask()
            self.net_model.dxdt = [self]

            self.params['epsilon'] = nn.parameter.Parameter(
                F.softplus(torch.ones((config.config["n_x"]),
                                      dtype=torch.float32, requires_grad=True)))
            self.params['alpha'] = nn.parameter.Parameter(
                F.softplus(torch.ones((config.config["n_x"]),
                                      dtype=torch.float32, requires_grad=True)))


            # self.params['debug'] = nn.parameter.Parameter(
            #     F.softplus(torch.ones((config.config["n_x"]),(config.config["n_x"]),
            #                dtype=torch.float32, requires_grad=True))
            # )

            print(len([a for a in self.parameters()]))

            self.hyper_params = {
                'l1': 1e-4,
                'l2': 1e-4
            }

            self.mse = torch.nn.MSELoss()

            self.stages = config.config['stages']

        def forward(self, t0, x, arg_dict):
            ''' recieve permutation with kwargs'''
            temp = self.net_model(x) + arg_dict['mu']
            temp = torch.tanh(temp)
            temp = self.params['epsilon'] * temp
            temp = temp - self.params['alpha'] * x
            return temp

        # let user define loss function
        def loss_fn(self, output, target):
            l1 = torch.sum(torch.abs(self.net_model.params['W']))
            l2 = torch.sum(torch.pow(self.net_model.params['W'], 2))

            # print(l1)
            # print(output.shape , target.shape) # debug
            mse_loss = self.mse(torch.squeeze(output), torch.squeeze(target))
            # print(mse_loss)

            total_loss = self.hyper_params['l1'] * l1 + self.hyper_params['l1'] * l2 + mse_loss
            # print(total_loss) # check that only the first time loss gets calculated

            return total_loss

        def on_after_backward(self):
            self.net_model.load_mask()
            return

        def training_epoch_end(self, outputs):
            cur_epoch = self.master[0].master[0].current_epoch

            if not self.stages:
                # no staging
                return
            else:
                idx_stage, change_hype = self.epoch2stage_idx(cur_epoch)

                stage = self.stages[idx_stage]
                if change_hype:
                    lr = stage['lr_val'] if 'lr_val' in stage else self.config.config['default']['lr_val']
                    l1 = stage['l1'] if 'l1' in stage else self.config.config['default']['l1']
                    l2 = stage['l2'] if 'l2' in stage else self.config.config['default']['l2']
                    nt = stage['nT'] if 'nT' in stage else self.config.config['default']['nT']
                    print(' ')
                    print('Changing Hyperparameters')
                    print(f'lr {lr}, l1 {l1}, l2 {l2}, nt {nt}')
                    self.master[0].master[0].lr = lr
                    self.hyper_params['l1'] = l1
                    self.hyper_params['l2'] = l2
                    self.master[0].nt = nt

                    self.master[0].master[0].configure_optimizers()

                else:
                    pass


class GeneralModel(pl.LightningModule):
    def __init__(self, config, dt):
        super().__init__()

        self.ode_solver = None

        self.dt = dt
        self.nt = 200

        self.nx = 0  # number of nodes
        self.config = config

        self.lit_module = None


    @staticmethod
    def build_model(config, args, dxdt: GeneralODE):
        gm = GeneralModel(config, args.dt)

        gm.add_module('dxdt', dxdt)
        dxdt.general_model = [gm]

        # set parameters of model
        # gm.params = dxdt.get_param_dict() # get them from dxdt

        # set ode_solver
        gm.ode_solver = get_ode_solver(args.ode_solver)
        return gm

    def get_device(self):
        return self.lit_module.get_device()

    # callbacks from torch lightning
    def loss_fn(self, output, target):
        return self.dxdt.loss_fn(output, target)

    def training_epoch_end(self, outputs):
        self.dxdt.training_epoch_end(outputs)

    def on_after_backward(self):
        self.dxdt.on_after_backward()

    def forward(self, t, y0, arg_dict):
        if self.nt==0:
            raise Exception('Number of iteration nt on model is zero')

        batch_size = y0.shape[0]

        # f, dt, nt, t0, y0, b_i
        ys = self.ode_solver.general_solve(
            ode=self.dxdt,
            dt=self.dt,
            nt=self.nt,
            t0=t,
            x=y0,
            arg_dict=arg_dict
            )

        # Take the last value of ys
        yhat = ys[-1].reshape(batch_size, -1, 1)

        # implement convergence
        # mean, sd = tf.nn.moments(ys, axes=0)
        #
        # dxdt = self._dxdt(ys[-1], u_t)
        # # [n_x, batch_size] for last ODE step
        # convergence_metric = tf.concat([mean, sd, dxdt], axis=0)

        convergence_metric = 0
        return yhat
        # return convergence_metric, yhat

class GeneralModel_v2(nn.Module):
    def __init__(self, config, dt, ud_dxdt, master):
        super().__init__()
        # initialize user defined module
        self.master = [master]
        self.dxdt = ud_dxdt(config, master=master)

        self.ode_solver = None

        self.dt = dt
        self.nt = 200

        self.nx = 0  # number of nodes
        self.config = config


    @staticmethod
    def build_model(config, args, user_defined_dxdt, master):
        gm = GeneralModel_v2(config, args.dt, user_defined_dxdt, master)

        gm.add_module('dxdt', gm.dxdt)
        gm.dxdt.master = [gm]

        # set parameters of model
        # gm.params = dxdt.get_param_dict() # get them from dxdt

        # set ode_solver
        gm.ode_solver = get_ode_solver(args.ode_solver)
        return gm

    def call_master(self):
        return self.master[0]

    def get_device(self):
        return self.call_master().get_device()

    # callbacks from torch lightning
    def loss_fn(self, output, target):
        return self.dxdt.loss_fn(output, target)

    def training_epoch_end(self, outputs):
        self.dxdt.training_epoch_end(outputs)

    def on_after_backward(self):
        if hasattr(self.dxdt, 'on_after_backward') and callable(getattr(self.dxdt, 'on_after_backward')):
            self.dxdt.on_after_backward()

    def forward(self, t, y0, arg_dict):
        if self.nt==0:
            raise Exception('Number of iteration nt on model is zero')

        batch_size = y0.shape[0]

        # f, dt, nt, t0, y0, b_i
        ys = self.ode_solver.general_solve(
            ode=self.dxdt,
            dt=self.dt,
            nt=self.nt,
            t0=t,
            x=y0,
            arg_dict=arg_dict
            )

        # Take the last value of ys
        yhat = ys[-1].reshape(batch_size, -1, 1)

        # implement convergence
        # mean, sd = tf.nn.moments(ys, axes=0)
        #
        # dxdt = self._dxdt(ys[-1], u_t)
        # # [n_x, batch_size] for last ODE step
        # convergence_metric = tf.concat([mean, sd, dxdt], axis=0)

        convergence_metric = 0
        return yhat
        # return convergence_metric, yhat

class LitSfeno_v2(pl.LightningModule):
    def __init__(self, config, args, ud_class_nm):
        super().__init__()

        # doesn't recieve model but creates
        self.model = GeneralModel_v2.build_model(config=config,
                                                 args=args,
                                                 user_defined_dxdt=ud_class_nm,
                                                 master=self)

        self.lr = 0.001
        self.l1_weight = 0
        self.l2_weight = 0
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def forward(self, t, y0, arg_dict):
        return self.model(self, t, y0, arg_dict)

    def get_device(self):
        return self.device

    def training_step(self, batch, batch_idx):
        inputs, targets = batch

        # input exists of initial state y0 and perterbation u
        y0, additional_args = inputs
        self.optimizer.zero_grad()

        # compute output
        outputs = self.model(t=0, y0=y0, arg_dict=additional_args)

        loss = self.loss_fn(outputs, targets)
        self.log("train_loss", loss,on_step=False, on_epoch=True)

        return loss

    # training_step has to be written to run DataParallel?

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch

        # input exists of initial state y0 and perterbation u
        y0, additional_args = inputs
        self.optimizer.zero_grad()

        # compute output
        outputs = self.model(t=0, y0=y0, arg_dict=additional_args)

        loss = self.loss_fn(outputs, targets)
        self.log("val_loss", loss, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        inputs, targets = batch

        # input exists of initial state y0 and perterbation u
        y0, additional_args = inputs
        self.optimizer.zero_grad()

        # compute output
        outputs = self.model(t=0, y0=y0, arg_dict=additional_args)

        loss = self.loss_fn(outputs, targets)
        self.log("test_loss", loss, on_step=False, on_epoch=True)

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        inputs, targets = batch
        print(targets.shape)
        y0, additional_args = inputs
        outputs = self.model(t=0, y0=y0, arg_dict=additional_args)
        return outputs, targets

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def loss_fn(self, output, target):
        return self.model.loss_fn(output, target)

    def training_epoch_end(self, outputs):
        # self.log("val_loss", outputs.mean(), on_step=False, on_epoch=True)
        if hasattr(self.model, 'training_epoch_end') and callable(getattr(self.model, 'training_epoch_end')):
            self.model.training_epoch_end(outputs)

    def validation_epoch_end(self, outputs):
        # self.log("val_loss", loss, on_step=False, on_epoch=True)
        if hasattr(self.model, 'validation_epoch_end') and callable(getattr(self.model, 'validation_epoch_end')):
            self.model.validation_epoch_end(outputs)

    def test_epoch_end(self, outputs):
        # self.log("val_loss", loss, on_step=False, on_epoch=True)
        if hasattr(self.model, 'test_epoch_end') and callable(getattr(self.model, 'test_epoch_end')):
            self.model.test_epoch_end(outputs)

    def on_after_backward(self):
        self.model.on_after_backward()

# class LitTrainer(pl.LightningModule):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model
#
#         self.lr = 0.001
#         self.l1_weight = 0
#         self.l2_weight = 0
#
#     def training_step(self,batch, batch_idx):
#         inputs, targets = batch
#
#         # input exists of initial state y0 and perterbation u
#         y0, additional_args = inputs
#         self.optimizer.zero_grad()
#
#         # compute output
#         outputs = self.model(t=0, y0=y0, arg_dict=additional_args)
#
#         loss = self.loss_fn(outputs, targets)
#         return loss
#
#     # training_step has to be written to run DataParallel?
#
#
#     def validation_step(self, batch, batch_idx):
#         inputs, targets = batch
#
#         # input exists of initial state y0 and perterbation u
#         y0, additional_args = inputs
#         self.optimizer.zero_grad()
#
#         # compute output
#         outputs = self.model(t=0, y0=y0, arg_dict=additional_args)
#
#         loss = self.loss_fn(outputs, targets)
#         self.log("val_loss", loss)
#
#         return loss
#
#     def validation_step(self, batch, batch_idx):
#         inputs, targets = batch
#
#         # input exists of initial state y0 and perterbation u
#         y0, additional_args = inputs
#         self.optimizer.zero_grad()
#
#         # compute output
#         outputs = self.model(t=0, y0=y0, arg_dict=additional_args)
#
#         loss = self.loss_fn(outputs, targets)
#         self.log("test_loss", loss)
#
#         return loss
#
#     def predict_step(self, batch, batch_idx, dataloader_idx=0):
#         x, y = batch
#         y_hat = self.model(x)
#         return y_hat
#
#     def configure_optimizers(self):
#         return torch.optim.Adam(self.model.parameters(),lr=self.lr)
#
#     def set_hyper_params(self,lr, l1, l2):
#         # for scheduling
#         self.lr = lr
#         self.l1_weight = l1
#         self.l2_weight = l2
#
#     def loss_fn(self, output, target):
#         l1 = 0 # torch.sum(torch.abs(self.model.params['W']))
#         l2 = 0 # torch.sum(torch.pow(self.model.params['W'],2))
#
#         # print(l1)
#         # print(output.shape , target.shape) # debug
#         mse_loss = self.mse(torch.squeeze(output), torch.squeeze(target))
#         # print(mse_loss)
#
#         total_loss = self.l1_weight * l1 + self.l2_weight * l2 + mse_loss
#         # print(total_loss) # check that only the first time loss gets calculated
#
#         return total_loss