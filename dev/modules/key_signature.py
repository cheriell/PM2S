import pytorch_lightning as pl
import os, sys
sys.path.insert(0, os.path.join(sys.path[0], '..'))
import torch.nn.functional as F
import torch.nn as nn

from pm2s.models.key_signature import RNNKeySignatureModel
from modules.utils import *

class KeySignatureModule(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = RNNKeySignatureModel()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return configure_optimizers(self)

    def configure_callbacks(self):
        return configure_callbacks(monitor='val_f1')

    def training_step(self, batch, batch_size):
        # Data
        x, y, length = batch
        x = x.float()
        y = y.long()
        length = length.long()

        # Forward pass
        y_hat = self(x)

        # Mask out the padding part
        mask = torch.ones(y_hat.shape).to(y_hat.device)
        for i in range(y_hat.shape[0]):
            mask[i, length[i]:] = 0
        y_hat = y_hat * mask

        # Loss
        loss = nn.NLLLoss()(y_hat, y)

        # Logging
        logs = {
            'train_loss': loss,
        }
        self.log_dict(logs, prog_bar=True)

        return {'loss': loss, 'logs': logs}


    def validation_step(self, batch, batch_size):
        # Data
        x, y, length = batch
        x = x.float()
        y = y.long()
        length = length.long()

        # Forward pass
        y_hat = self(x)

        # Mask out the padding part
        for i in range(y_hat.shape[0]):
            y_hat[i, length[i]:] = 0

        # Loss
        loss = nn.NLLLoss()(y_hat, y)

        # Metrics
        prec_macro_all, rec_macro_all, f_macro_all = 0, 0, 0
        prec_weighted_all, rec_weighted_all, f_weighted_all = 0, 0, 0

        for i in range(y_hat.shape[0]):
            # get sample from batch
            y_hat_i = y_hat[i, :, :length[i]].topk(1, dim=0)[1][0]
            y_i = y[i, :length[i]]

            # get accuracies
            (
                prec_macro, rec_macro, f_macro,
                prec_weighted, rec_weighted, f_weighted
            ) = classification_report_framewise(y_i, y_hat_i)

            prec_macro_all += prec_macro
            rec_macro_all += rec_macro
            f_macro_all += f_macro
            prec_weighted_all += prec_weighted
            rec_weighted_all += rec_weighted
            f_weighted_all += f_weighted

        prec_macro_all /= y_hat.shape[0]
        rec_macro_all /= y_hat.shape[0]
        f_macro_all /= y_hat.shape[0]
        prec_weighted_all /= y_hat.shape[0]
        rec_weighted_all /= y_hat.shape[0]
        f_weighted_all /= y_hat.shape[0]

        # Logging
        logs = {
            'val_loss': loss,
            'val_prec_macro': prec_macro_all,
            'val_rec_macro': rec_macro_all,
            'val_f_macro': f_macro_all,
            'val_prec_weighted': prec_weighted_all,
            'val_rec_weighted': rec_weighted_all,
            'val_f_weighted': f_weighted_all,
            'val_f1': f_macro_all,
        }
        self.log_dict(logs, prog_bar=True)

        return {'loss': loss, 'logs': logs}