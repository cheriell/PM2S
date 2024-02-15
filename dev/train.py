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
from configs import training_configs


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
    data_module = Pm2sDataModule(args)

    # Model
    if args.feature == 'beat':
        model = BeatModule(omit_input_feature=args.omit_input_feature)
    elif args.feature == 'quantisation':
        model = QuantisationModule(beat_type=args.beat_type)
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
        gpus=training_configs[args.feature]['gpus'],
    )

    # Train
    trainer.fit(model, data_module)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train a model.')

    ####################################################
    # Required arguments
    ####################################################
    # Paths to the workspace and the datasets
    parser.add_argument('--workspace', type=str, help='Workspace directory.')
    parser.add_argument('--ASAP', type=str, help='ASAP dataset directory.')
    parser.add_argument('--A_MAPS', type=str, help='A_MAPS dataset directory.')
    parser.add_argument('--CPM', type=str, help='CPM dataset directory.')
    # Output feature type
    parser.add_argument('--feature', type=str, help='Feature type. (beat, quantisation, hand_part, key_signature, time_signature)')

    ####################################################
    # Optional arguments for specific models
    ####################################################
    # For Beat Tracking (RNNJointBeatModel)
    # For training models to be tested on transcribed MIDIs
    parser.add_argument('--mode', type=str, default=None, help='Mode of training for the transcribed MIDIs. (clean, transcribed, mixed.) None for training and evaluation on the clean performance MIDIs.')
    # For input feature ablation study
    parser.add_argument('--omit_input_feature', type=str, default=None, help='Omit input feature for ablation study. (pitch, onset, duration, velocity.) None for no ablation study.')

    # For Quantisation (RNNJointQuantisationModel)
    parser.add_argument('--beat_type', type=str, default='ground_truth', help='Type of beat to be used for training. (ground_truth, estimated, mixed)')

    ####################################################
    # For traing with the whole dataset
    ####################################################
    parser.add_argument('--full_train', action='store_true', help='Training with the whole dataset or not (i.e. only the training set).')

    args = parser.parse_args()

    train(args)
