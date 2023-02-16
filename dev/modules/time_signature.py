import pytorch_lightning as pl
import os, sys
sys.path.insert(0, os.path.join(sys.path[0], '..'))
import torch.nn.functional as F
import torch.nn as nn

from pm2s.models.time_signature import RNNTimeSignatureModel
from modules.utils import *

class TimeSignatureModule(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = RNNTimeSignatureModel()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return configure_optimizers(self)

    def configure_callbacks(self):
        return configure_callbacks(monitor='val_f1')

    def training_step(self, batch, batch_size):
        # Data
        x, y_tn, y_td, length = batch
        x = x.float()
        y_tn = y_tn.long()
        y_td = y_td.long()
        length = length.long()

        # Forward pass
        y_tn_hat, y_td_hat = self(x)

        # Mask out the padding part
        pad_mask = torch.ones((y_tn_hat.shape[0], y_tn_hat.shape[2])).to(y_tn_hat.device)
        for i in range(y_tn_hat.shape[0]):
            pad_mask[i, length[i]:] = 0
        y_tn_hat = y_tn_hat * pad_mask.unsqueeze(1)
        y_td_hat = y_td_hat * pad_mask.unsqueeze(1)

        # Loss
        loss_tn = nn.NLLLoss(ignore_index=0)(y_tn_hat, y_tn)
        loss_td = nn.NLLLoss(ignore_index=0)(y_td_hat, y_td)
        loss = loss_tn + loss_td

        # Logging
        logs = {
            'train_loss': loss,
            'train_loss_tn': loss_tn,
            'train_loss_td': loss_td,
        }
        self.log_dict(logs, prog_bar=True)

        return {'loss': loss, 'logs': logs}

    def validation_step(self, batch, batch_size):
        # Data
        x, y_tn, y_td, length = batch
        x = x.float()
        y_tn = y_tn.long()
        y_td = y_td.long()
        length = length.long()

        # Forward pass
        y_tn_hat, y_td_hat = self(x)

        # Mask out the padding part
        for i in range(y_tn_hat.shape[0]):
            y_tn_hat[i, :, length[i]:] = 0
            y_td_hat[i, :, length[i]:] = 0

        # Loss
        loss_tn = nn.NLLLoss(ignore_index=0)(y_tn_hat, y_tn)
        loss_td = nn.NLLLoss(ignore_index=0)(y_td_hat, y_td)
        loss = loss_tn + loss_td

        # Metrics
        precs_macro_tn, recs_macro_tn, fs_macro_tn = 0, 0, 0
        precs_weighted_tn, recs_weighted_tn, fs_weighted_tn = 0, 0, 0

        precs_macro_td, recs_macro_td, fs_macro_td = 0, 0, 0
        precs_weighted_td, recs_weighted_td, fs_weighted_td = 0, 0, 0

        for i in range(x.shape[0]):
            # get sample from batch
            y_tn_hat_i = y_tn_hat[i, :, :length[i]].topk(1, dim=0)[1][0]
            y_td_hat_i = y_td_hat[i, :, :length[i]].topk(1, dim=0)[1][0]
            y_tn_i = y_tn[i, :length[i]]
            y_td_i = y_td[i, :length[i]]

            # filter out ignored indexes (the same as padding)
            y_tn_hat_i = y_tn_hat_i[y_tn_i != 0]
            y_tn_i = y_tn_i[y_tn_i != 0]
            y_td_hat_i = y_td_hat_i[y_td_i != 0]
            y_td_i = y_td_i[y_td_i != 0]

            # get accuracies
            (
                prec_macro_tn, rec_macro_tn, f_macro_tn,
                prec_weighted_tn, rec_weighted_tn, f_weighted_tn
            ) = classification_report_framewise(y_tn_i, y_tn_hat_i)
            (
                prec_macro_td, rec_macro_td, f_macro_td,
                prec_weighted_td, rec_weighted_td, f_weighted_td
            ) = classification_report_framewise(y_td_i, y_td_hat_i)
            
            precs_macro_tn += prec_macro_tn
            recs_macro_tn += rec_macro_tn
            fs_macro_tn += f_macro_tn
            precs_weighted_tn += prec_weighted_tn
            recs_weighted_tn += rec_weighted_tn
            fs_weighted_tn += f_weighted_tn

            precs_macro_td += prec_macro_td
            recs_macro_td += rec_macro_td
            fs_macro_td += f_macro_td
            precs_weighted_td += prec_weighted_td
            recs_weighted_td += rec_weighted_td
            fs_weighted_td += f_weighted_td

        precs_macro_tn /= x.shape[0]
        recs_macro_tn /= x.shape[0]
        fs_macro_tn /= x.shape[0]
        precs_weighted_tn /= x.shape[0]
        recs_weighted_tn /= x.shape[0]
        fs_weighted_tn /= x.shape[0]

        precs_macro_td /= x.shape[0]
        recs_macro_td /= x.shape[0]
        fs_macro_td /= x.shape[0]
        precs_weighted_td /= x.shape[0]
        recs_weighted_td /= x.shape[0]
        fs_weighted_td /= x.shape[0]

        # Logging
        logs = {
            'val_loss': loss,
            'val_loss_tn': loss_tn,
            'val_loss_td': loss_td,
            'val_f1_tn': fs_macro_tn,
            'val_f1_td': fs_macro_td,
            'val_f1': (fs_macro_tn + fs_macro_td) / 2,
        }
        self.log_dict(logs, prog_bar=True)

        return {'loss': loss, 'logs': logs}






