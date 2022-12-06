import pytorch_lightning as pl
import os, sys
sys.path.insert(0, os.path.join(sys.path[0], '..'))
import torch.nn.functional as F
import torch.nn as nn

from pm2s.models.beat import RNNJointBeatModel
from modules.utils import *

class BeatModule(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = RNNJointBeatModel()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return configure_optimizers(self)

    def configure_callbacks(self):
        return configure_callbacks(monitor='val_f_b')

    def training_step(self, batch, batch_idx):
        # Data
        x, y_b, y_db, y_ibi, length = batch
        x = x.float()
        y_b = y_b.float()
        y_db = y_db.float()
        y_ibi = y_ibi.long()
        length = length.long()

        # Forward pass
        y_b_hat, y_db_hat, y_ibi_hat = self(x)

        # Mask out the padding part
        mask = torch.ones(y_b_hat.shape).to(y_b_hat.device)
        for i in range(y_b_hat.shape[0]):
            mask[i, length[i]:] = 0
        y_b_hat = y_b_hat * mask
        y_db_hat = y_db_hat * mask
        y_ibi_hat = y_ibi_hat * mask.unsqueeze(1)

        # Loss
        loss_b = F.binary_cross_entropy(y_b_hat, y_b)
        loss_db = F.binary_cross_entropy(y_db_hat, y_db)
        loss_ibi = nn.NLLLoss(ignore_index=0)(y_ibi_hat, y_ibi)
        loss = loss_b + loss_db + loss_ibi

        # Logging
        logs = {
            'train_loss': loss,
            'train_loss_b': loss_b,
            'train_loss_db': loss_db,
            'train_loss_ibi': loss_ibi,
        }
        self.log_dict(logs, prog_bar=True)

        return {'loss': loss, 'logs': logs}

    def validation_step(self, batch, batch_idx):
        # Data
        x, y_b, y_db, y_ibi, length = batch
        x = x.float()
        y_b = y_b.float()
        y_db = y_db.float()
        y_ibi = y_ibi.long()
        length = length.long()

        # Forward pass
        y_b_hat, y_db_hat, y_ibi_hat = self(x)

        # Mask out the padding part
        for i in range(y_b_hat.shape[0]):
            y_b_hat[i, length[i]:] = 0
            y_db_hat[i, length[i]:] = 0
            y_ibi_hat[i, :, length[i]:] = 0

        # Loss
        loss_b = F.binary_cross_entropy(y_b_hat, y_b)
        loss_db = F.binary_cross_entropy(y_db_hat, y_db)
        loss_ibi = nn.NLLLoss(ignore_index=0)(y_ibi_hat, y_ibi)
        loss = loss_b + loss_db + loss_ibi

        # Metrics
        accs_b, precs_b, recs_b, fs_b = 0, 0, 0, 0
        accs_db, precs_db, recs_db, fs_db = 0, 0, 0, 0

        for i in range(x.shape[0]):
            # get sample from batch
            y_b_hat_i = torch.round(y_b_hat[i, :length[i]])
            y_db_hat_i = torch.round(y_db_hat[i, :length[i]])
            y_ibi_hat_i = y_ibi_hat[i, :, :length[i]].topk(1, dim=0)[1][0]
            
            y_b_i = y_b[i, :length[i]]
            y_db_i = y_db[i, :length[i]]
            y_ibi_i = y_ibi[i, :length[i]]

            # filter out ignore indexes
            y_ibi_hat_i = y_ibi_hat_i[y_ibi_i != 0]
            y_ibi_i = y_ibi_i[y_ibi_i != 0]

            # get accuracy
            acc_b, prec_b, rec_b, f_b = f_measure_framewise(y_b_i, y_b_hat_i)
            acc_db, prec_db, rec_db, f_db = f_measure_framewise(y_db_i, y_db_hat_i)
            
            accs_b += acc_b
            precs_b += prec_b
            recs_b += rec_b
            fs_b += f_b

            accs_db += acc_db
            precs_db += prec_db
            recs_db += rec_db
            fs_db += f_db

        accs_b /= x.shape[0]
        precs_b /= x.shape[0]
        recs_b /= x.shape[0]
        fs_b /= x.shape[0]

        accs_db /= x.shape[0]
        precs_db /= x.shape[0]
        recs_db /= x.shape[0]
        fs_db /= x.shape[0]

        # Logging
        logs = {
            'val_loss': loss,
            'val_loss_b': loss_b,
            'val_loss_db': loss_db,
            'val_loss_ibi': loss_ibi,
            'val_acc_b': accs_b,
            'val_prec_b': precs_b,
            'val_rec_b': recs_b,
            'val_f_b': fs_b,
            'val_acc_db': accs_db,
            'val_prec_db': precs_db,
            'val_rec_db': recs_db,
            'val_f_db': fs_db,
            'val_f1': fs_b,  # this will be used as the monitor for logging and checkpointing callbacks
        }
        self.log_dict(logs, prog_bar=True)

        return {'val_loss': loss, 'logs': logs}
