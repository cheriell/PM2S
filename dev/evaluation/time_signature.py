import numpy as np
import torch

from data.dataset_time_signature import TimeSignatureDataset
from pm2s.models.time_signature import CNNTimeSignatureModel
from modules.utils import f_measure_framewise


def evaluate_time_signature_prediction(args):

    testset = TimeSignatureDataset(
        workspace=args.workspace, 
        split='test',
    )
    model = CNNTimeSignatureModel()
    model.load_state_dict(torch.load(args.model_state_dict_path))
    model = model.to(args.device)
    model.eval()

    targets = []
    predictions = []

    for i in range(len(testset)):
        print('testing sample {}/{}'.format(i+1, len(testset)), end='\r')

        # Get data
        note_sequence, time_sigs, length = testset[i]
        
        # Get prediction by majority voting
        note_sequence = torch.Tensor(note_sequence).to(args.device).float().unsqueeze(0)
        ts_probs = model(note_sequence)
        ts_probs = ts_probs.squeeze(0).detach().cpu().numpy()  # (seq_len,)
        ts_pred = (ts_probs[:length] > 0.5).astype(int)
        ts_pred = np.bincount(ts_pred).argmax()

        # Get ground truth by majority votings
        ts_targ = np.bincount(time_sigs[:length].astype(int)).argmax()

        # Append
        predictions.append(ts_pred)
        targets.append(ts_targ)

    print()

    # Compute f-measure
    predictions = np.array(predictions)
    targets = np.array(targets)
    print('predictions:\n', predictions)
    print('targets:\n', targets)
    acc, p, r, f = f_measure_framewise(torch.Tensor(targets), torch.Tensor(predictions))
    print('acc: {:.4f}, p: {:.4f}, r: {:.4f}, f: {:.4f}'.format(acc, p, r, f))
