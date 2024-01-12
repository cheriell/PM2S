import torch

from data.dataset_beat import BeatDataset
from pm2s.models.beat import RNNJointBeatModel
from modules.utils import f_measure_framewise

beat_prob_threshold = 0.5

def evaluate_beat_prediction(args):

    testset = BeatDataset(
        workspace=args.workspace, 
        split='test',
    )
    model = RNNJointBeatModel()
    model.load_state_dict(torch.load(args.model_state_dict_path))
    model = model.to(args.device)
    model.eval()

    all_beat_metrics = []
    all_downbeat_metrics = []

    for i in range(len(testset)):
        print('testing sample {}/{}'.format(i+1, len(testset)), end='\r')

        # Get data
        note_sequence, beat_gt, downbeat_gt, _, length = testset[i]

        # Get note-level prediction
        note_sequence = torch.Tensor(note_sequence).to(args.device).float().unsqueeze(0)
        beat_probs, downbeat_probs, _ = model(note_sequence)
        beat_probs = beat_probs.squeeze(0).detach().cpu().numpy()
        downbeat_probs = downbeat_probs.squeeze(0).detach().cpu().numpy()
        beat_pred = (beat_probs > beat_prob_threshold).astype(int)
        downbeat_pred = (downbeat_probs > beat_prob_threshold).astype(int)

        # Evaluate
        beat_metrics = f_measure_framewise(torch.Tensor(beat_gt), torch.Tensor(beat_pred))
        downbeat_metrics = f_measure_framewise(torch.Tensor(downbeat_gt), torch.Tensor(downbeat_pred))

        # Append
        all_beat_metrics.append(torch.Tensor(beat_metrics))
        all_downbeat_metrics.append(torch.Tensor(downbeat_metrics))

    print()

    # Compute average
    all_beat_metrics = torch.stack(all_beat_metrics)
    all_downbeat_metrics = torch.stack(all_downbeat_metrics)
    ave_beat_metrics = torch.mean(all_beat_metrics, dim=0)
    ave_downbeat_metrics = torch.mean(all_downbeat_metrics, dim=0)

    # Print
    print('ave_beat_metrics:\n acc: {:.4f}, p: {:.4f}, r: {:.4f}, f: {:.4f}'.format(
        ave_beat_metrics[0], ave_beat_metrics[1], ave_beat_metrics[2], ave_beat_metrics[3]))
    print('ave_downbeat_metrics:\n acc: {:.4f}, p: {:.4f}, r: {:.4f}, f: {:.4f}'.format(
        ave_downbeat_metrics[0], ave_downbeat_metrics[1], ave_downbeat_metrics[2], ave_downbeat_metrics[3]))

