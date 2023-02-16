import pytorch_lightning as pl
import os, sys
sys.path.insert(0, os.path.join(sys.path[0], '..'))
import torch.nn.functional as F
import torch.nn as nn

from pm2s.models.hand_part import RNNHandPartModel
from modules.utils import *

class HandPartModule(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = RNNHandPartModel()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return configure_optimizers(self)

    def configure_callbacks(self):
        return configure_callbacks(monitor='val_f1')

    def training_step(self, batch, batch_size):
        # Data
        x, y, y_mask, length = batch
        x = x.float()
        y = y.float()
        y_mask = y_mask.float()
        length = length.long()

        # Forward pass
        y_hat = self(x)

        # Mask out the padding part & mask from data loader
        pad_mask = torch.ones(y_hat.shape).to(y_hat.device)
        for i in range(y_hat.shape[0]):
            pad_mask[i, length[i]:] = 0
        y_hat = y_hat * pad_mask
        y_hat = y_hat * y_mask

        # Loss
        loss = F.binary_cross_entropy(y_hat, y)

        # Logging
        logs = {
            'train_loss': loss,
        }
        self.log_dict(logs, prog_bar=True)

        return {'loss': loss, 'logs': logs}

    def validation_step(self, batch, batch_size):
        # Data
        x, y, y_mask, length = batch
        x = x.float()
        y = y.float()
        y_mask = y_mask.float()
        length = length.long()

        # Forward pass
        y_hat = self(x)

        # Mask out the padding part & mask from data loader
        for i in range(y_hat.shape[0]):
            y_hat[i, length[i]:] = 0
        y_hat = y_hat * y_mask

        # Loss
        loss = F.binary_cross_entropy(y_hat, y)

        # Metrics
        accs, precs, recs, f1s = 0, 0, 0, 0

        for i in range(y_hat.shape[0]):
            # get sample from batch
            y_hat_i = torch.round(y_hat[i, :length[i]])
            y_i = y[i, :length[i]]

            # accuracies
            acc, prec, rec, f1 = f_measure_framewise(y_i, y_hat_i)

            accs += acc
            precs += prec
            recs += rec
            f1s += f1

        accs /= x.shape[0]
        precs /= x.shape[0]
        recs /= x.shape[0]
        f1s /= x.shape[0]

        # Logging
        logs = {
            'val_loss': loss,
            'val_acc': accs,
            'val_prec': precs,
            'val_rec': recs,
            'val_f1': f1s,
        }
        self.log_dict(logs, prog_bar=True)

        return {'loss': loss, 'logs': logs}


        