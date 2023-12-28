import pytorch_lightning as pl
import os, sys
sys.path.insert(0, os.path.join(sys.path[0], '..'))
import torch
import torch.nn.functional as F

from pm2s.models.time_signature import CNNTimeSignatureModel
from modules.utils import configure_optimizers, configure_callbacks, f_measure_framewise

class TimeSignatureModule(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = CNNTimeSignatureModel()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return configure_optimizers(self)

    def configure_callbacks(self):
        return configure_callbacks(monitor='val_f1')

    def training_step(self, batch, batch_size):
        # Data
        x, y_ts, length = batch
        x = x.float()
        y_ts = y_ts.float()
        length = length.long()

        # Forward pass
        y_ts_hat = self(x)  # (batch_size, max_length)

        # Mask out the padding part
        pad_mask = torch.ones((y_ts_hat.shape[0], y_ts_hat.shape[1])).to(y_ts_hat.device)
        for i in range(y_ts_hat.shape[0]):
            pad_mask[i, length[i]:] = 0
        y_ts_hat = y_ts_hat * pad_mask

        # Loss
        loss = F.binary_cross_entropy(y_ts_hat, y_ts)

        # Metrics
        accs_ts, precs_ts, recs_ts, fs_ts = 0, 0, 0, 0

        for i in range(x.shape[0]):
            # get sample from batch
            y_ts_hat_i = torch.round(y_ts_hat[i, :length[i]])
            y_ts_i = y_ts[i, :length[i]]

            # filter out ignored indexes (the same as padding)
            # No need to filter out ignored indexes for binary classification

            # get accuracies
            acc_ts, prec_ts, rec_ts, f_ts = f_measure_framewise(y_ts_i, y_ts_hat_i)

            accs_ts += acc_ts
            precs_ts += prec_ts
            recs_ts += rec_ts
            fs_ts += f_ts

        accs_ts /= x.shape[0]
        precs_ts /= x.shape[0]
        recs_ts /= x.shape[0]
        fs_ts /= x.shape[0]

        # Logging
        logs = {
            'train_loss': loss,
            'train_acc': accs_ts,
            'train_prec': precs_ts,
            'train_rec': recs_ts,
            'train_f1': fs_ts,
        }
        self.log_dict(logs, prog_bar=True)

        return {'loss': loss, 'logs': logs}

    def validation_step(self, batch, batch_size):
        # Data
        x, y_ts, length = batch
        x = x.float()
        y_ts = y_ts.float()
        length = length.long()

        # Forward pass
        y_ts_hat = self(x)  # (batch_size, max_length)

        # Mask out the padding part
        for i in range(y_ts_hat.shape[0]):
            y_ts_hat[i, length[i]:] = 0

        # Loss
        loss = F.binary_cross_entropy(y_ts_hat, y_ts)

        # Metrics
        accs_ts, precs_ts, recs_ts, fs_ts = 0, 0, 0, 0

        for i in range(x.shape[0]):
            # get sample from batch
            y_ts_hat_i = torch.round(y_ts_hat[i, :length[i]])
            y_ts_i = y_ts[i, :length[i]]
            if i == 0:
                print(y_ts_hat_i)
                print(y_ts_i)
                print()

            # filter out ignored indexes (the same as padding)
            # No need to filter out ignored indexes for binary classification

            # get accuracies
            acc_ts, prec_ts, rec_ts, f_ts = f_measure_framewise(y_ts_i, y_ts_hat_i)
            
            accs_ts += acc_ts
            precs_ts += prec_ts
            recs_ts += rec_ts
            fs_ts += f_ts

        accs_ts /= x.shape[0]
        precs_ts /= x.shape[0]
        recs_ts /= x.shape[0]
        fs_ts /= x.shape[0]

        # Logging
        logs = {
            'val_loss': loss,
            'val_acc': accs_ts,
            'val_prec': precs_ts,
            'val_rec': recs_ts,
            'val_f1': fs_ts,
        }
        self.log_dict(logs, prog_bar=True)

        return {'loss': loss, 'logs': logs}






