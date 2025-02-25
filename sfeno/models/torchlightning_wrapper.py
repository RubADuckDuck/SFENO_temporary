import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

class LitSfeno(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.lit_module = [self]

        self.lr = 0.001
        self.l1_weight = 0
        self.l2_weight = 0
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.lr)


    def forward(self, t, y0, arg_dict):
        return self.model(self, t, y0, arg_dict)

    def training_step(self,batch, batch_idx):
        inputs, targets = batch

        # input exists of initial state y0 and perterbation u
        y0, additional_args = inputs
        self.optimizer.zero_grad()

        # compute output
        outputs = self.model(t=0, y0=y0, arg_dict=additional_args)

        loss = self.loss_fn(outputs, targets)
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
        self.log("val_loss", loss)

        return loss

    def test_step(self, batch, batch_idx):
        inputs, targets = batch

        # input exists of initial state y0 and perterbation u
        y0, additional_args = inputs
        self.optimizer.zero_grad()

        # compute output
        outputs = self.model(t=0, y0=y0, arg_dict=additional_args)

        loss = self.loss_fn(outputs, targets)
        self.log("test_loss", loss)

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self.model(x)
        return y_hat

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(),lr=self.lr)

    def loss_fn(self, output, target):
        return self.model.loss_fn(output, target)

    def training_epoch_end(self, outputs):
        self.model.training_epoch_end(outputs)

    def on_after_backward(self):
        self.model.on_after_backward()


