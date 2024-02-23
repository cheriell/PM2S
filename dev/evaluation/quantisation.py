import torch

from data.dataset_quantisation import QuantisationDataset
from pm2s.models.quantisation import RNNJointQuantisationModel
from modules.utils import classification_report_framewise


def evaluate_quantisation_prediction(args):

    testset = QuantisationDataset(
        workspace=args.workspace,
        split='test',
        mode=None,
    )
    model = RNNJointQuantisationModel()
    model.load_state_dict(torch.load(args.model_state_dict_path))
    model = model.to(args.device)
    model.eval()

    all_onset_metrics = []
    all_value_metrics = []

    for i in range(len(testset)):
        print('testing sample {}/{}'.format(i+1, len(testset)), end='\r')

        # Get data
        note_sequence, _, onsets, note_values, length = testset[i]

        # Get prediction
        note_sequence = torch.Tensor(note_sequence).to(args.device).float().unsqueeze(0)
        probs_onset, probs_value = model(note_sequence)
        
        # Post-processing
        onset_positions_idx = probs_onset[0].topk(1, dim=0)[1].squeeze(0) # (n_notes,)
        note_values_idx = probs_value[0].topk(1, dim=0)[1].squeeze(0) # (n_notes,)

        onset_positions_idx = onset_positions_idx.detach().cpu().numpy()
        note_values_idx = note_values_idx.detach().cpu().numpy()

        # Evaluate
        onsets_idx_targ = onsets[:length]
        values_idx_targ = note_values[:length]
        onsets_idx_pred = onset_positions_idx[:length]
        values_idx_pred = note_values_idx[:length]

        metrics_onset = classification_report_framewise(onsets_idx_targ, onsets_idx_pred)
        metrics_value = classification_report_framewise(values_idx_targ, values_idx_pred)
        all_onset_metrics.append(torch.Tensor(metrics_onset))
        all_value_metrics.append(torch.Tensor(metrics_value))

    print()

    # Compute average
    all_onset_metrics = torch.stack(all_onset_metrics)
    all_value_metrics = torch.stack(all_value_metrics)
    ave_onset_metrics = torch.mean(all_onset_metrics, dim=0)
    ave_value_metrics = torch.mean(all_value_metrics, dim=0)

    # Print
    print('ave_onset_metrics:\n prec_macro_onset: {:.4f}\t rec_macro_onset: {:.4f}\t f1_macro_onset: {:.4f}\t prec_weighted_onset: {:.4f}\t rec_weighted_onset: {:.4f}\t f1_weighted_onset: {:.4f}'.format(ave_onset_metrics[0], ave_onset_metrics[1], ave_onset_metrics[2], ave_onset_metrics[3], ave_onset_metrics[4], ave_onset_metrics[5]))
    print('ave_value_metrics:\n prec_macro_value: {:.4f}\t rec_macro_value: {:.4f}\t f1_macro_value: {:.4f}\t prec_weighted_value: {:.4f}\t rec_weighted_value: {:.4f}\t f1_weighted_value: {:.4f}'.format(ave_value_metrics[0], ave_value_metrics[1], ave_value_metrics[2], ave_value_metrics[3], ave_value_metrics[4], ave_value_metrics[5]))
