import torch

from data.dataset_key_signature import KeySignatureDataset
from pm2s.models.key_signature import RNNKeySignatureModel
from modules.utils import classification_report_framewise

def evaluate_key_signature_prediction(args):

    testset = KeySignatureDataset(
        workspace=args.workspace, 
        split='test',
    )
    model = RNNKeySignatureModel()
    model.load_state_dict(torch.load(args.model_state_dict_path))
    model = model.to(args.device)
    model.eval()

    all_metrics = []

    for i in range(len(testset)):
        print('testing sample {}/{}'.format(i+1, len(testset)), end='\r')

        # Get data
        note_sequence, key_targ, length = testset[i]

        # Get note-level prediction
        note_sequence = torch.Tensor(note_sequence).to(args.device).float().unsqueeze(0)
        key_probs = model(note_sequence)
        key_pred = key_probs[0].topk(1, dim=0)[1].squeeze(0).cpu().detach().numpy() # (seq_len,)

        # Evaluate
        metrics = classification_report_framewise(torch.Tensor(key_targ), torch.Tensor(key_pred))

        # Append
        all_metrics.append(torch.Tensor(metrics))

    print()

    # Compute average
    all_metrics = torch.stack(all_metrics)
    ave_metrics = torch.mean(all_metrics, dim=0)

    # Print
    print('ave_metrics:\n prec_macro: {:.4f}, rec_macro: {:.4f}, f1_macro: {:.4f}, prec_weighted: {:.4f}, rec_weighted: {:.4f}, f1_weighted: {:.4f}'.format(ave_metrics[0], ave_metrics[1], ave_metrics[2], ave_metrics[3], ave_metrics[4], ave_metrics[5]))


