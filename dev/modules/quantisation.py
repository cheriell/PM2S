import pytorch_lightning as pl
import os, sys
sys.path.insert(0, os.path.join(sys.path[0], '..'))
import torch.nn.functional as F
import torch.nn as nn

from pm2s.models.quantisation import RNNJointQuantisationModel
from modules.utils import *

class QuantisationModule(pl.LightningModule):

    def __init__(self, beat_model_checkpoint="../_model_state_dicts/beat/RNNJointBeatModel.pth"):
        super().__init__()
        self.model = RNNJointQuantisationModel(beat_model_checkpoint=beat_model_checkpoint)

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
        mask = torch.ones(y_o_hat.shape[0], y_o_hat.shape[2]).to(y_o_hat.device)
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

        # Forward passclear
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
        prec_macro_o_all, rec_macro_o_all, f_macro_o_all = 0, 0, 0
        prec_weighted_o_all, rec_weighted_o_all, f_weighted_o_all = 0, 0, 0
        count_o = 0

        prec_macro_n_all, rec_macro_n_all, f_macro_n_all = 0, 0, 0
        prec_weighted_n_all, rec_weighted_n_all, f_weighted_n_all = 0, 0, 0
        count_n = 0

        for i in range(x.shape[0]):
            # get sample from batch
            y_o_hat_i = y_o_hat[i, :, :length[i]].topk(1, dim=0)[1][0]
            y_n_hat_i = y_n_hat[i, :, :length[i]].topk(1, dim=0)[1][0]

            y_o_i = y_o[i, :length[i]]
            y_n_i = y_n[i, :length[i]]

            # filter out ignored indexes
            y_n_hat_i = y_n_hat_i[y_n_i != 0]
            y_n_i = y_n_i[y_n_i != 0]

            # get accuracy
            (
                prec_macro_o, rec_macro_o, f_macro_o,
                prec_weighted_o, rec_weighted_o, f_weighted_o,
            ) = classification_report_framewise(y_o_i, y_o_hat_i)
            (
                prec_macro_n, rec_macro_n, f_macro_n,
                prec_weighted_n, rec_weighted_n, f_weighted_n,
            ) = classification_report_framewise(y_n_i, y_n_hat_i)

            if y_o_i.shape[0] > 0:
                prec_macro_o_all += prec_macro_o
                rec_macro_o_all += rec_macro_o
                f_macro_o_all += f_macro_o
                prec_weighted_o_all += prec_weighted_o
                rec_weighted_o_all += rec_weighted_o
                f_weighted_o_all += f_weighted_o
                count_o += 1

            if y_n_i.shape[0] > 0:
                prec_macro_n_all += prec_macro_n
                rec_macro_n_all += rec_macro_n
                f_macro_n_all += f_macro_n
                prec_weighted_n_all += prec_weighted_n
                rec_weighted_n_all += rec_weighted_n
                f_weighted_n_all += f_weighted_n
                count_n += 1

        if count_o > 0:
            prec_macro_o_all /= count_o
            rec_macro_o_all /= count_o
            f_macro_o_all /= count_o
            prec_weighted_o_all /= count_o
            rec_weighted_o_all /= count_o
            f_weighted_o_all /= count_o

        if count_n > 0:
            prec_macro_n_all /= count_n
            rec_macro_n_all /= count_n
            f_macro_n_all /= count_n
            prec_weighted_n_all /= count_n
            rec_weighted_n_all /= count_n
            f_weighted_n_all /= count_n

        # Logging
        logs = {
            'val_loss': loss,
            'val_loss_o': loss_o,
            'val_loss_n': loss_n,
            'val_f_macro_o': f_macro_o_all,
            'val_f_weighted_o': f_weighted_o_all,
            'val_f_macro_n': f_macro_n_all,
            'val_f_weighted_n': f_weighted_n_all,
            'val_f1': f_weighted_o_all,  # for logging and checkpointing
        }
        self.log_dict(logs, prog_bar=True)

        return {'val_loss': loss, 'logs': logs}

            




        