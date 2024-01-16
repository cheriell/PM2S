import torch

from data.dataset_hand_part import HandPartDataset
from pm2s.models.hand_part import RNNHandPartModel
from modules.utils import f_measure_framewise

def evaluate_hand_part_prediction(args):
    
    testset = HandPartDataset(
        workspace=args.workspace, 
        split='test',
    )
    model = RNNHandPartModel()
    model.load_state_dict(torch.load(args.model_state_dict_path))
    model = model.to(args.device)
    model.eval()

    all_metrics = []

    for i in range(len(testset)):
        print('testing sample {}/{}'.format(i+1, len(testset)), end='\r')

        # Get data
        note_sequence, hand_part_targ, _, length = testset[i]

        # Get note-level prediction
        note_sequence = torch.Tensor(note_sequence).to(args.device).float().unsqueeze(0)
        hand_part_probs = model(note_sequence)
        hand_part_probs = hand_part_probs.squeeze(0).detach().cpu().numpy()
        hand_part_pred = (hand_part_probs > 0.5).astype(int)

        # Evaluate
        metrics = f_measure_framewise(torch.Tensor(hand_part_targ), torch.Tensor(hand_part_pred))

        # Append
        all_metrics.append(torch.Tensor(metrics))

    print()

    # Compute average
    all_metrics = torch.stack(all_metrics)
    ave_metrics = torch.mean(all_metrics, dim=0)

    # Print
    print('ave_metrics:\n acc: {:.4f}, p: {:.4f}, r: {:.4f}, f: {:.4f}'.format(
        ave_metrics[0], ave_metrics[1], ave_metrics[2], ave_metrics[3]))