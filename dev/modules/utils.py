import torch
import pytorch_lightning as pl
import numpy as np
from sklearn.metrics import classification_report


def configure_optimizers(module, lr=1e-3, step_size=50):
    optimizer = torch.optim.AdamW(
        module.parameters(),
        lr=lr,
        betas=(0.8, 0.8),
        eps=1e-4,
        weight_decay=1e-2,
    )
    scheduler_lrdecay = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size,
        gamma=0.1
    )
    return [optimizer], [scheduler_lrdecay]

def configure_callbacks(monitor='val_f1', mode='max'):
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor=monitor,
        mode=mode,
        save_top_k=3,
        filename='{epoch}-{val_loss:.2f}-{val_f1:.2f}',
        save_last=True,
    )
    earlystop_callback = pl.callbacks.EarlyStopping(
        monitor=monitor,
        patience=200,
        mode=mode,
    )
    return [checkpoint_callback, earlystop_callback]


def f_measure_framewise(y, y_hat):
    acc = (y_hat == y).float().mean()
    TP = torch.logical_and(y_hat==1, y==1).float().sum()
    FP = torch.logical_and(y_hat==1, y==0).float().sum()
    FN = torch.logical_and(y_hat==0, y==1).float().sum()

    p = TP / (TP + FP + np.finfo(float).eps)
    r = TP / (TP + FN + np.finfo(float).eps)
    f = 2 * p * r / (p + r + np.finfo(float).eps)
    return acc, p, r, f

def classification_report_framewise(y, y_hat):
    report = classification_report(y.tolist(), y_hat.tolist(), output_dict=True)

    prec_macro = report['macro avg']['precision']
    rec_macro = report['macro avg']['recall']
    f1_macro = report['macro avg']['f1-score']
    prec_weighted = report['weighted avg']['precision']
    rec_weighted = report['weighted avg']['recall']
    f1_weighted = report['weighted avg']['f1-score']
    
    return prec_macro, rec_macro, f1_macro, prec_weighted, rec_weighted, f1_weighted