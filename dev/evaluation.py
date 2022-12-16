import argparse
import sys, os
sys.path.insert(0, os.path.join(sys.path[0], '../'))
import pandas as pd
import pretty_midi as pm


def evaluate_beat(args):
    from pm2s.features.beat import RNNJointBeatProcessor
    import mir_eval

    processor = RNNJointBeatProcessor(model_state_dict_path='../_model_state_dicts/beat/RNNJointBeatModel.pth')

    metadata = pd.read_csv('metadata/metadata.csv')
    metadata_test = metadata[metadata['split'] == 'test']

    f1_all = []

    for _, row in metadata_test.iterrows():
        midi_file = os.path.join(args.ACPAS, row['midi_perfm'])

        # Process the MIDI recording to get the beat predictions
        beats_pred = processor.process(midi_file)

        # Ground truth beats
        midi_data = pm.PrettyMIDI(midi_file)
        beats_targ = midi_data.get_beats()

        # evaluate
        beats_pred = mir_eval.beat.trim_beats(beats_pred)
        beats_targ = mir_eval.beat.trim_beats(beats_targ)
        f1 = mir_eval.beat.f_measure(beats_targ, beats_pred)

        f1_all.append(f1)
        print('F1 score for the current performance: {:.3f}, Mean F1 score: {:.3f}'.format(f1, sum(f1_all)/len(f1_all)))



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train a model.')

    parser.add_argument('--ACPAS', type=str, help='Path to the ACPAS dataset.')
    parser.add_argument('--feature', type=str, help='Feature type.')

    parser.add_argument('--model_state_dict_path_beat', type=str, default='../_model_state_dicts/beat/RNNJointBeatModel.pth', help='Path to the model state dict.')

    args = parser.parse_args()

    evaluate_beat(args)
