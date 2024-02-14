import warnings
warnings.filterwarnings('ignore')
import argparse
import torch
from pathlib import Path
import os

def save_model(args):

    # Load pytorch lightning module
    if args.feature == 'beat':
        from modules.beat import BeatModule
        module = BeatModule.load_from_checkpoint(args.model_checkpoint_path, omit_input_feature=args.omit_input_feature)

    elif args.feature == 'quantisation':
        from modules.quantisation import QuantisationModule
        module = QuantisationModule.load_from_checkpoint(args.model_checkpoint_path)
        
    elif args.feature == 'hand_part':
        from modules.hand_part import HandPartModule
        module = HandPartModule.load_from_checkpoint(args.model_checkpoint_path)

    elif args.feature == 'time_signature':
        from modules.time_signature import TimeSignatureModule
        module = TimeSignatureModule.load_from_checkpoint(args.model_checkpoint_path)

    elif args.feature == 'key_signature':
        from modules.key_signature import KeySignatureModule
        module = KeySignatureModule.load_from_checkpoint(args.model_checkpoint_path)

    else:
        raise ValueError('Invalid feature type.')
        
    # Save model
    Path.mkdir(Path(args.model_state_dict_path).parent, parents=True, exist_ok=True)
    torch.save(module.model.state_dict(), args.model_state_dict_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Save a model as state dict.')

    parser.add_argument('--feature', type=str, help='Feature type.')
    parser.add_argument('--omit_input_feature', type=str, default=None, help='Omit input feature for ablation study. (pitch, onset, duration, velocity)')

    parser.add_argument('--model_checkpoint_path', type=str, help='Path to model checkpoint.')
    parser.add_argument('--model_state_dict_path', type=str, default='_model_state_dict.pth', help='Path to save the model state dict.')

    args = parser.parse_args()

    save_model(args)