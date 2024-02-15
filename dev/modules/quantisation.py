import pytorch_lightning as pl
import os, sys
sys.path.insert(0, os.path.join(sys.path[0], '..'))
import torch.nn.functional as F
import torch.nn as nn

from pm2s.models.quantisation import RNNJointQuantisationModel
from modules.utils import *

class QuantisationModule(pl.LightningModule):

    def __init__(self, beat_type='estimated'):
        super().__init__()
        self.model = RNNJointQuantisationModel(beat_type=beat_type)

    def forward(self, x, x_beat):
        return self.model(x, x_beat)

    def configure_optimizers(self):
        return configure_optimizers(self)

    def configure_callbacks(self):
        return configure_callbacks(monitor='val_f1')

    def training_step(self, batch, batch_size):
        # Data
        x, x_beat, y_onset, y_value, length = batch
        x = x.float()
        x_beat = x_beat.float()
        y_onset = y_onset.long()
        y_value = y_value.long()
        length = length.long()

        # Forward pass
        y_onset_hat, y_value_hat = self(x, x_beat)

        # Mask out the padding part
        mask = torch.ones(y_onset_hat.shape[0], y_onset_hat.shape[2]).to(y_onset_hat.device)
        for i in range(y_onset_hat.shape[0]):
            mask[i, length[i]:] = 0
        y_onset_hat = y_onset_hat * mask.unsqueeze(1)
        y_value_hat = y_value_hat * mask.unsqueeze(1)

        # Loss
        loss_onset = nn.NLLLoss()(y_onset_hat, y_onset)
        loss_value = nn.NLLLoss(ignore_index=0)(y_value_hat, y_value)
        loss = loss_onset + loss_value

        # Logging
        logs = {
            'train_loss': loss,
            'train_loss_onset': loss_onset,
            'train_loss_value': loss_value,
        }
        self.log_dict(logs, prog_bar=True)

        for param in self.model.beat_model.parameters():
            print(param[:5,0,0,0])
            break

        return {'loss': loss, 'logs': logs}
        

    def validation_step(self, batch, batch_size):
        # Data
        x, x_beat, y_onset, y_value, length = batch
        x = x.float()
        x_beat = x_beat.float()
        y_onset = y_onset.long()
        y_value = y_value.long()
        length = length.long()

        # Forward passclear
        y_onset_hat, y_value_hat = self(x, x_beat=None)

        # Mask out the padding part
        for i in range(y_onset_hat.shape[0]):
            y_onset_hat[i, :, length[i]:] = 0
            y_value_hat[i, :, length[i]:] = 0

        # Loss
        loss_onset = nn.NLLLoss()(y_onset_hat, y_onset)
        loss_value = nn.NLLLoss(ignore_index=0)(y_value_hat, y_value)
        loss = loss_onset + loss_value

        # Metrics
        prec_macro_onset_all, rec_macro_onset_all, f_macro_onset_all = 0, 0, 0
        prec_weighted_onset_all, rec_weighted_onset_all, f_weighted_onset_all = 0, 0, 0
        count_onset = 0

        prec_macro_value_all, rec_macro_value_all, f_macro_value_all = 0, 0, 0
        prec_weighted_value_all, rec_weighted_value_all, f_weighted_value_all = 0, 0, 0
        count_value = 0

        for i in range(x.shape[0]):
            # get sample from batch
            y_onset_hat_i = y_onset_hat[i, :, :length[i]].topk(1, dim=0)[1][0]
            y_value_hat_i = y_value_hat[i, :, :length[i]].topk(1, dim=0)[1][0]

            y_onset_i = y_onset[i, :length[i]]
            y_value_i = y_value[i, :length[i]]

            # filter out ignored indexes
            y_value_hat_i = y_value_hat_i[y_value_i != 0]
            y_value_i = y_value_i[y_value_i != 0]

            # get accuracy
            (
                prec_macro_o, rec_macro_o, f_macro_o,
                prec_weighted_o, rec_weighted_o, f_weighted_o,
            ) = classification_report_framewise(y_onset_i, y_onset_hat_i)
            (
                prec_macro_n, rec_macro_n, f_macro_n,
                prec_weighted_n, rec_weighted_n, f_weighted_n,
            ) = classification_report_framewise(y_value_i, y_value_hat_i)

            if y_onset_i.shape[0] > 0:
                prec_macro_onset_all += prec_macro_o
                rec_macro_onset_all += rec_macro_o
                f_macro_onset_all += f_macro_o
                prec_weighted_onset_all += prec_weighted_o
                rec_weighted_onset_all += rec_weighted_o
                f_weighted_onset_all += f_weighted_o
                count_onset += 1

            if y_value_i.shape[0] > 0:
                prec_macro_value_all += prec_macro_n
                rec_macro_value_all += rec_macro_n
                f_macro_value_all += f_macro_n
                prec_weighted_value_all += prec_weighted_n
                rec_weighted_value_all += rec_weighted_n
                f_weighted_value_all += f_weighted_n
                count_value += 1

        if count_onset > 0:
            prec_macro_onset_all /= count_onset
            rec_macro_onset_all /= count_onset
            f_macro_onset_all /= count_onset
            prec_weighted_onset_all /= count_onset
            rec_weighted_onset_all /= count_onset
            f_weighted_onset_all /= count_onset

        if count_value > 0:
            prec_macro_value_all /= count_value
            rec_macro_value_all /= count_value
            f_macro_value_all /= count_value
            prec_weighted_value_all /= count_value
            rec_weighted_value_all /= count_value
            f_weighted_value_all /= count_value

        # Logging
        logs = {
            'val_loss': loss,
            'val_loss_onset': loss_onset,
            'val_loss_value': loss_value,
            'val_f_macro_onset': f_macro_onset_all,
            'val_f_weighted_onset': f_weighted_onset_all,
            'val_f_macro_value': f_macro_value_all,
            'val_f_weighted_value': f_weighted_value_all,
            'val_f1': f_weighted_onset_all,  # for logging and checkpointing
        }
        self.log_dict(logs, prog_bar=True)

        return {'val_loss': loss, 'logs': logs}

            




        