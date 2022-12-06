import pytorch_lightning as pl
import os, sys
sys.path.insert(0, os.path.join(sys.path[0], '..'))
import torch.nn.functional as F
import torch.nn as nn

from pm2s.models.quantisation import RNNJointQuantisationModel
from modules.utils import *

class QuantisationModule(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = RNNJointQuantisationModel()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return configure_optimizers(self)

    def configure_callbacks(self):
        return configure_callbacks(monitor='val_f1')

    def training_step(self, batch, batch_size):
        # Data
        x, y_o, y_n, length = batch
        x = x.float()
        y_o = y_o.long()
        y_n = y_n.long()
        length = length.long()

        # Forward pass
        _, _, y_o_hat, y_n_hat = self(x)

        # Mask out the padding part
        mask = torch.ones(y_o_hat.shape).to(y_o_hat.device)
        for i in range(y_o_hat.shape[0]):
            mask[i, length[i]:] = 0
        y_o_hat = y_o_hat * mask.unsqueeze(1)
        y_n_hat = y_n_hat * mask.unsqueeze(1)

        # Loss
        loss_o = nn.NLLLoss()(y_o_hat, y_o)
        loss_n = nn.NLLLoss(ignore_index=0)(y_n_hat, y_n)
        loss = loss_o + loss_n

        # Logging
        logs = {
            'train_loss': loss,
            'train_loss_o': loss_o,
            'train_loss_n': loss_n,
        }
        self.log_dict(logs, prog_bar=True)

        return {'loss': loss, 'logs': logs}
        

    def validation_step(self, batch, batch_size):
        # Data
        x, y_o, y_n, length = batch
        x = x.float()
        y_o = y_o.long()
        y_n = y_n.long()
        length = length.long()

        # Forward pass
        _, _, y_o_hat, y_n_hat = self(x)

        # Mask out the padding part
        for i in range(y_o_hat.shape[0]):
            y_o_hat[i, :, length[i]:] = 0
            y_n_hat[i, :, length[i]:] = 0

        # Loss
        loss_o = nn.NLLLoss()(y_o_hat, y_o)
        loss_n = nn.NLLLoss(ignore_index=0)(y_n_hat, y_n)
        loss = loss_o + loss_n

        # Metrics
        


        