import torch
import mir_eval

from data.dataset_beat import BeatDataset
from pm2s.models.beat import RNNJointBeatModel
from pm2s.features.beat import RNNJointBeatProcessor
from modules.utils import f_measure_framewise

beat_prob_threshold = 0.5

def evaluate_beat_prediction(args):

    testset = BeatDataset(
        workspace=args.workspace, 
        split='test',
        mode=args.mode,
    )
    model = RNNJointBeatModel(omit_input_feature=args.omit_input_feature)
    model.load_state_dict(torch.load(args.model_state_dict_path))
    model = model.to(args.device)
    model.eval()

    all_beat_metrics_note_level = []
    all_downbeat_metrics_note_level = []
    all_beat_f1 = []
    all_downbeat_f1 = []

    for i in range(len(testset)):
        print('testing sample {}/{}'.format(i+1, len(testset)), end='\r')

        # Get data
        note_sequence, beat_gt, downbeat_gt, _, length = testset[i]

        # Get note-level prediction
        x = torch.Tensor(note_sequence).to(args.device).float().unsqueeze(0)
        beat_probs, downbeat_probs, _ = model(x)
        beat_probs = beat_probs.squeeze(0).detach().cpu().numpy()
        downbeat_probs = downbeat_probs.squeeze(0).detach().cpu().numpy()

        # Note-level prediction
        beat_pred = (beat_probs > beat_prob_threshold).astype(int)
        downbeat_pred = (downbeat_probs > beat_prob_threshold).astype(int)

        # Post-processing
        onsets = note_sequence[:, 1]
        beats_final_pred, downbeats_final_pred = RNNJointBeatProcessor.pps(beat_probs, downbeat_probs, onsets)

        # Evaluation

        # Note-level
        beat_metrics_note_level = f_measure_framewise(torch.Tensor(beat_gt), torch.Tensor(beat_pred))
        downbeat_metrics_note_level = f_measure_framewise(torch.Tensor(downbeat_gt), torch.Tensor(downbeat_pred))

        # After post-processing
        beats_final_targ, downbeats_final_targ = testset.__get_beat_annotations__(i)

        beats_final_pred_trimmed = mir_eval.beat.trim_beats(beats_final_pred)
        beats_final_targ_trimmed = mir_eval.beat.trim_beats(beats_final_targ)
        downbeats_final_pred_trimmed = mir_eval.beat.trim_beats(downbeats_final_pred)
        downbeats_final_targ_trimmed = mir_eval.beat.trim_beats(downbeats_final_targ)

        f1_beats = mir_eval.beat.f_measure(beats_final_targ_trimmed, beats_final_pred_trimmed)
        f1_downbeats = mir_eval.beat.f_measure(downbeats_final_targ_trimmed, downbeats_final_pred_trimmed)

        # Append to results
        all_beat_metrics_note_level.append(torch.Tensor(beat_metrics_note_level))
        all_downbeat_metrics_note_level.append(torch.Tensor(downbeat_metrics_note_level))

        all_beat_f1.append(f1_beats)
        all_downbeat_f1.append(f1_downbeats)

    print()

    # Compute average
    all_beat_metrics_note_level = torch.stack(all_beat_metrics_note_level)
    all_downbeat_metrics_note_level = torch.stack(all_downbeat_metrics_note_level)
    ave_beat_metrics_note_level = torch.mean(all_beat_metrics_note_level, dim=0)
    ave_downbeat_metrics_note_level = torch.mean(all_downbeat_metrics_note_level, dim=0)

    ave_beat_f1 = sum(all_beat_f1) / len(all_beat_f1)
    ave_downbeat_f1 = sum(all_downbeat_f1) / len(all_downbeat_f1)

    # Print
    print('------------------------------------')
    print('Note-level beat and downbeat metrics:')
    print('ave_beat_metrics:\n acc: {:.4f}, p: {:.4f}, r: {:.4f}, f: {:.4f}'.format(
        ave_beat_metrics_note_level[0], ave_beat_metrics_note_level[1], ave_beat_metrics_note_level[2], ave_beat_metrics_note_level[3]))
    print('ave_downbeat_metrics:\n acc: {:.4f}, p: {:.4f}, r: {:.4f}, f: {:.4f}'.format(
        ave_downbeat_metrics_note_level[0], ave_downbeat_metrics_note_level[1], ave_downbeat_metrics_note_level[2], ave_downbeat_metrics_note_level[3]))
    print('------------------------------------')
    print('Beat and downbeat metrics after post-processing:')
    print('ave_beat_f1: {:.4f}'.format(ave_beat_f1))
    print('ave_downbeat_f1: {:.4f}'.format(ave_downbeat_f1))


