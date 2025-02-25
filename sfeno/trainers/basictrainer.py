# import sfeno.models
import math

import numpy as np
import pandas as pd
import torch

from time import time

class BasicTrainer:
    def __init__(self,
                 model,
                 trainloader,
                 validloader,
                 config):

        self.config = config

        self.model = model
        self.mse = torch.nn.MSELoss()
        # print(model.params.items())
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        self.l1_weight = 1
        self.l2_weight = 1

        self.train_loader = trainloader
        self.validation_loader = validloader

        self.report_term = 500

        self.epoch_num = 0

        # track substage for file naming
        self.cur_substage_name = ''

        # track substage number
        self.i_substage = 0

        self.timer = TraingTimer()
        self.timer.start_timer()

    def train_stages(self,stages, detect_anomaly=False):
        ''' follows through 'stages' to train the given model
        '''

        if detect_anomaly:
            with torch.autograd.detect_anomaly():
                for st_idx, cur_stg in enumerate(stages):
                    cur_sub_stges = cur_stg['sub_stages']
                    cur_nt = cur_stg['nT']

                    print(f"Stage {st_idx} ==============================")
                    print(f'  nT: {cur_nt}')

                    # cellbox
                    self.model.nt = cur_nt

                    self.train_by_ss(cur_stg)
        else:
            for st_idx, cur_stg in enumerate(stages):
                cur_sub_stges = cur_stg['sub_stages']
                cur_nt = cur_stg['nT']

                print(f"Stage {st_idx} ==============================")
                print(f'  nT: {cur_nt}')

                # cellbox
                self.model.nt = cur_nt
                self.train_by_ss(cur_stg)



    def set_hyper_params(self,lr, l1, l2):
        # for scheduling
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.l1_weight = l1
        self.l2_weight = l2

    def loss_fn(self, output, target):
        l1 = 0 # torch.sum(torch.abs(self.model.params['W']))
        l2 = 0 # torch.sum(torch.pow(self.model.params['W'],2))

        # print(l1)
        # print(output.shape , target.shape) # debug
        mse_loss = self.mse(torch.squeeze(output), torch.squeeze(target))
        # print(mse_loss)

        total_loss = self.l1_weight * l1 + self.l2_weight * l2 + mse_loss
        # print(total_loss) # check that only the first time loss gets calculated

        return total_loss

    def train_epoch(self):
        running_loss = 0
        last_loss = 0
        avg_loss = 0

        for idx_batch, data in enumerate(self.train_loader):
            inputs, targets = data

            # input exists of initial state y0 and perterbation u
            y0, additional_args = inputs
            self.optimizer.zero_grad()

            # compute output
            outputs = self.model(t=0, y0=y0, arg_dict=additional_args)

            # calc loss
            loss = self.loss_fn(outputs, targets)
            loss.backward()

            # # debuging step
            # self.model._dxdt.check_param()
            # self.model._dxdt.check_param_grad()

            # Adjust weight
            self.optimizer.step()


            # report
            running_loss += loss.item()
            avg_loss += loss.item()
            if idx_batch % self.report_term == self.report_term - 1:
                last_loss = running_loss / self.report_term

                print(f' batch {idx_batch+1} loss {last_loss}')
                running_loss = 0

        avg = avg_loss / (idx_batch + 1)
        return avg_loss, last_loss

    def optimizer_step(self):
        self.optimizer.step()

        self.run_step_callbacks()
        return

    def run_step_callbacks(self):
        for cb in self.step_callback_ls:
            cb()
        return

    def validate(self):
        running_vloss = 0
        last_vloss = 0
        avg_vloss = 0

        for i, vdata in enumerate(self.validation_loader):
            inputs, targets = vdata

            # input exists of initial state y0 and perterbation u
            y0, additional_args = inputs

            # compute output
            outputs = self.model(t=0, y0=y0, arg_dict=additional_args)

            loss = self.loss_fn(outputs, targets)

            running_vloss += loss.item()
            avg_vloss += loss.item()
            if i % self.report_term == self.report_term - 1:
                last_loss = running_vloss / self.report_term

                print(f' batch {i+1} loss {last_loss}')
                running_loss = 0

        avg_vloss = avg_vloss / (i + 1)

        return avg_vloss, last_vloss

    def train(self, epochs):
        best_vloss = math.inf
        for epoch in range(epochs):
            print(f'Epoch {self.epoch_num + 1}')

            self.model.train(True)
            avg_loss, last_loss = self.train_epoch()

            self.model.train(False)

            running_vloss = 0.0
            for i, vdata in enumerate(self.validation_loader):
                vinputs, vx_targets = vdata
                voutputs = self.model(*vinputs)
                vloss = self.loss_fn(voutputs, vx_targets)
                running_vloss += vloss

            avg_vloss = running_vloss / (i+1)
            print(f'Loss train {avg_loss} valid {avg_vloss}')

            # track best
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = f'model_{self.epoch_num+1}'
                torch.save(self.model.state_dict(), model_path)

            # debuging step
            self.model._dxdt.check_param()
            self.model._dxdt.check_param_grad()

            self.epoch_num += 1

    def train_by_ss(self, stage):
        '''
        unpacks infomation from config and organizes, then runs given set of substages in order
        '''

        # time logger
        nt = stage['nT']
        substages = stage['sub_stages']
        for substage in substages:
            print(substage)
            n_iter_buffer = substage['n_iter_buffer'] if 'n_iter_buffer' in substage else self.config.config['n_iter_buffer']
            n_iter = substage['n_iter'] if 'n_iter' in substage else self.config.config['n_iter']
            n_iter_patience = substage['n_iter_patience'] if 'n_iter_patience' in substage else self.config.config['n_iter_patience']
            n_epoch = substage['n_epoch'] if 'n_epoch' in substage else self.config.config['n_epoch']
            l1 = substage['l1lambda'] if 'l1lambda' in substage else self.config.config['l1lambda'] if 'l1lambda' in self.config.config else 0
            l2 = substage['l2lambda'] if 'l2lambda' in substage else self.config.config['l2lambda'] if 'l2lambda' in self.config.config else 0
            lr = substage['lr_val']



            print(f'n_iter_buffer: {n_iter_buffer}')
            print(f'n_iter: {n_iter}')
            print(f'n_iter_patience: {n_iter_patience}')
            print(f'n_epoch: {n_epoch}')
            print(f'l1: {l1}')
            print(f'l2: {l2}')
            print(f'lr: {lr}')
            print('===========================================================')
            self.cur_substage_name = f"substage_{self.i_substage}__" \
                                     f"nib_{n_iter_buffer}_" \
                                     f"ni_{n_iter}_" \
                                     f"nip_{n_iter_patience}_" \
                                     f"ne_{n_epoch}_" \
                                     f"l1_{l1}_" \
                                     f"l2_{l2}_" \
                                     f"lr_{lr}"
            # add timer checkpoint
            self.timer.add_ss_checkpoint()
            self.train_substage(lr, l1_lambda=l1, l2_lambda=l2, n_epoch=n_epoch,
                                n_iter=n_iter, n_iter_buffer=n_iter_buffer, n_iter_patience=n_iter_patience, stage=stage)

    def train_substage(self, lr, l1_lambda, l2_lambda, n_epoch, n_iter, n_iter_buffer, n_iter_patience, stage):
        """
        Training function that does single stage of training. The stage training can be repeated and modified to give better
        training result.
        Args:
            l1_lambda (float): l1 regularization weight
            l2_lambda (float): l2 regularization weight
            n_epoch (int): maximum number of epochs
            n_iter (int): maximum number of iterations
            n_iter_buffer (int): training loss moving average window
            n_iter_patience (int): training loss tolerance
            stage: stage configuration
        """
        # entering new substage
        self.i_substage += 1

        n_unchanged = 0
        idx_iter = 0

        # time per iteration
        tpe = 0

        best_params = TemporaryS()

        # important!!!!
        # should be able to apply lr, l1_lambda, l2_lambda
        self.set_hyper_params(lr=lr, l1=l1_lambda, l2=l2_lambda)

        for idx_epoch in range(n_epoch):
            print(f'Epoch {idx_epoch} running!')
            if idx_iter > n_iter or n_unchanged > n_iter_patience:
                break

            while True:
                # view time per iteration
                tpe = self.timer.get_elapsed_time()

                if idx_iter > n_iter or n_unchanged > n_iter_patience:
                    break

                self.model.train(True)
                avg_loss, last_loss = self.train_epoch()

                self.model.train(False)
                avg_vloss, last_vloss = self.validate()
                new_loss = avg_vloss


                # early stopping
                idx_iter += 1

                # debug
                print('-------------------------------------------------')
                print(f'time per iteration: {tpe}')
                print(f'{idx_iter}th iter loss : {new_loss}')

                if new_loss < best_params.loss_min: # problem is how do we calculate new loss?
                    # if the models loss have progress
                    # reset unchanged counter
                    n_unchanged = 0

                    # take a screenshot of the state of model
                    best_params.screenshot()
                else:
                    # else the model show no progress
                    n_unchanged += 1

                # record loss
                best_params.track_loss(new_loss)
        # save model
        self.save_model()
        # save loss to
        best_params.save_csv_to(f"{self.config.config['experiment_id']}/{self.cur_substage_name}.csv")

    def save_model(self):
        pass



class TemporaryS: # temporary screenshot
    def __init__(self):
        self.loss_min = math.inf
        self.loss_ls = []



    def track_loss(self, loss):
        self.loss_ls.append(loss)
        self.check_loss_min(loss)

    def check_loss_min(self,loss):
        if loss < self.loss_min:
            self.loss_min = loss
        return

    def save_csv_to(self, path):
        loss_arr = np.array(self.loss_ls)
        df = pd.DataFrame(loss_arr)

        df.to_csv(path)

    def screenshot(self): # not implemented
        pass

class TraingTimer:
    def __init__(self):
        self.start_time = None
        self.sub_stage_checkpoints = []
        self.prev_time = None

    def get_cur_time(self):
        return time()

    def start_timer(self):
        self.start_time = self.get_cur_time()
        self.prev_time = self.start_time

    def add_ss_checkpoint(self):
        self.sub_stage_checkpoints.append(self.get_cur_time())

    def get_elapsed_time(self):
        # returns duration between last call for this function
        ct = self.get_cur_time()

        et = ct - self.prev_time

        self.prev_time = ct

        return et

# class Screenshot(dict):
#     """summarize the model"""
#     def __init__(self, args, n_iter_buffer):
#         # initialize loss_min
#         super().__init__()
#         self.loss_min = 1000
#         # initialize tuning_metric
#         self.saved_losses = [self.loss_min]
#         self.n_iter_buffer = n_iter_buffer
#         # initialize verbose
#         self.summary = {}
#         self.summary = {}
#         self.substage_i = []
#         self.export_verbose = args.export_verbose
#
#     def avg_n_iters_loss(self, new_loss):
#         """average the last few losses"""
#         self.saved_losses = self.saved_losses + [new_loss]
#         self.saved_losses = self.saved_losses[-self.n_iter_buffer:]
#         return sum(self.saved_losses) / len(self.saved_losses)
#
#     def screenshot(self, sess, model, substage_i, node_index, loss_min, args):
#         """evaluate models"""
#         self.substage_i = substage_i
#         self.loss_min = loss_min
#         # Save the variables to disk.
#         if self.export_verbose > 0:
#             params = sess.run(model.params)
#             for item in params:
#                 try:
#                     params[item] = pd.DataFrame(params[item], index=node_index[0])
#                 except Exception:
#                     params[item] = pd.DataFrame(params[item])
#             self.update(params)
#
#         if self.export_verbose > 1 or self.export_verbose == -1:  # no params but y_hat
#             sess.run(model.iter_eval.initializer, feed_dict=model.args.feed_dicts['test_set'])
#             y_hat = eval_model(sess, model.iter_eval, model.eval_yhat, args.feed_dicts['test_set'], return_avg=False)
#             y_hat = pd.DataFrame(y_hat, columns=node_index[0])
#             self.update({'y_hat': y_hat})
#
#         if self.export_verbose > 2:
#             try:
#                 # TODO: not yet support data iterators
#                 summary_train = sess.run(model.convergence_metric,
#                                          feed_dict={model.in_pert: args.dataset['pert_train']})
#                 summary_test = sess.run(model.convergence_metric, feed_dict={model.in_pert: args.dataset['pert_test']})
#                 summary_valid = sess.run(model.convergence_metric,
#                                          feed_dict={model.in_pert: args.dataset['pert_valid']})
#                 summary_train = pd.DataFrame(summary_train, columns=[node_index.values + '_mean', node_index.values +
#                                                                      '_sd', node_index.values + '_dxdt'])
#                 summary_test = pd.DataFrame(summary_test, columns=[node_index.values + '_mean', node_index.values +
#                                                                    '_sd', node_index.values + '_dxdt'])
#                 summary_valid = pd.DataFrame(summary_valid, columns=[node_index.values + '_mean', node_index.values +
#                                                                      '_sd', node_index.values + '_dxdt'])
#                 self.update(
#                     {'summary_train': summary_train, 'summary_test': summary_test, 'summary_valid': summary_valid}
#                 )
#             except Exception:
#                 pass
#
#     def save(self):
#         """save model parameters"""
#         for file in glob.glob(str(self.substage_i) + "_best.*.csv"):
#             os.remove(file)
#         for key in self:
#             self[key].to_csv("{}_best.{}.loss.{}.csv".format(self.substage_i, key, self.loss_min))



# if __name__ == '__main__':
#     args = 0
#     model = model.factory(args)
#
#
#
#     t = BasicTrainer(model)

