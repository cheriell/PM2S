import warnings
warnings.filterwarnings('ignore')
import argparse
import pytorch_lightning as pl
pl.seed_everything(42)
import os

from data.data_module import Pm2sDataModule
from modules.beat import BeatModule
from modules.quantisation import QuantisationModule
from modules.hand_part import HandPartModule
from modules.key_signature import KeySignatureModule
from modules.time_signature import TimeSignatureModule
from configs import gpus


## -------------------------
## DEBUGGING BLOCK
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
torch.autograd.set_detect_anomaly(True)
## END DEBUGGING BLOCK
## -------------------------

# Set cuda visable devices
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'



def train(args):
    # Data
    data_module = Pm2sDataModule(args, feature=args.feature, full_train=args.full_train)

    # Model
    if args.feature == 'beat':
        model = BeatModule()
    elif args.feature == 'quantisation':
        model = QuantisationModule()
    elif args.feature == 'hand_part':
        model = HandPartModule()
    elif args.feature == 'key_signature':
        model = KeySignatureModule()
    elif args.feature == 'time_signature':
        model = TimeSignatureModule()
    else:
        raise ValueError('Invalid feature type.')

    # Logger
    logger = pl.loggers.MLFlowLogger(
        experiment_name='{}-training'.format(args.feature),
        tracking_uri=os.path.join(args.workspace, 'mlruns'),
        run_name='run-0',
    )

    # Trainer
    trainer = pl.Trainer(
        default_root_dir=os.path.join(args.workspace, 'mlruns'),
        logger=logger,
        log_every_n_steps=50,
        reload_dataloaders_every_n_epochs=True,
        gpus=gpus,
    )

    # Train
    trainer.fit(model, data_module)

    
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train a model.')

    parser.add_argument('--workspace', type=str, help='Workspace directory.')
    parser.add_argument('--ASAP', type=str, help='ASAP dataset directory.')
    parser.add_argument('--A_MAPS', type=str, help='A_MAPS dataset directory.')
    parser.add_argument('--CPM', type=str, help='CPM dataset directory.')
    parser.add_argument('--feature', type=str, help='Feature type.')
    parser.add_argument('--full_train', action='store_true', help='Training with the whole dataset or not (only the training set).')

    args = parser.parse_args()

    train(args)
