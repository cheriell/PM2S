import warnings
warnings.filterwarnings('ignore')
import argparse
import torch

def save_model(args):
    if args.feature == 'beat':
        from modules.beat import BeatModule
        module = BeatModule.load_from_checkpoint(args.model_checkpoint_path)
        
    torch.save(module.model.state_dict(), args.model_save_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Save a model as state dict.')

    parser.add_argument('--model_checkpoint_path', type=str, help='Path to model checkpoint.')
    parser.add_argument('--model_save_path', type=str, help='Path to save model state dict.')
    parser.add_argument('--feature', type=str, help='Feature type.')

    args = parser.parse_args()

    save_model(args)