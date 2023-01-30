import warnings
warnings.filterwarnings('ignore')
import argparse
import torch
from pathlib import Path

def save_model(args):
    if args.feature == 'beat':
        from modules.beat import BeatModule
        module = BeatModule.load_from_checkpoint(args.model_checkpoint_path)

        model_save_path = '../_model_state_dicts/beat/RNNJointBeatModel.pth'

    elif args.feature == 'quantisation':
        from modules.quantisation import QuantisationModule
        module = QuantisationModule.load_from_checkpoint(args.model_checkpoint_path, beat_model_checkpoint=args.beat_model_checkpoint)

        model_save_path = '../_model_state_dicts/quantisation/RNNJointQuantisationModel.pth'

    else:
        raise ValueError('Invalid feature type.')
        
    Path.mkdir(Path(model_save_path).parent, parents=True, exist_ok=True)
    torch.save(module.model.state_dict(), model_save_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Save a model as state dict.')

    parser.add_argument('--model_checkpoint_path', type=str, help='Path to model checkpoint.')
    parser.add_argument('--beat_model_checkpoint', type=str, help='Path to beat model checkpoint.', default='../_model_state_dicts/beat/RNNJointBeatModel.pth')
    parser.add_argument('--feature', type=str, help='Feature type.')

    args = parser.parse_args()

    save_model(args)