import torch
import numpy as np

from data.dataset_base import BaseDataset
from data.data_augmentation import DataAugmentation
from pm2s.constants import *
from pm2s.models.beat import RNNJointBeatModel
from configs import training_configs

class QuantisationDataset(BaseDataset):

    def __init__(self, workspace, split, mode):
        super().__init__(workspace, split, from_asap=False, feature='quantisation', no_transcribed=True, mode=mode)
        
        # Initialise data augmentation
        self.dataaug = DataAugmentation(feature='quantisation')

    def __getitem__(self, idx):

        row = self._sample_row(idx)
        note_sequence, annotations = self._load_data(row)

        # Get binarised beat probabilities at note onsets
        def get_x_beat_from_ground_truth():
            beats = annotations['beats']

            # time to beat/downbeat/inter-beat-interval dictionaries
            end_time = max(beats[-1], note_sequence[-1][1] + note_sequence[-1][2]) + 1.0
            time2beat = np.zeros(int(np.ceil(end_time / resolution)))
            for idx, beat in enumerate(beats):
                l = np.round((beat - tolerance) / resolution).astype(int)
                r = np.round((beat + tolerance) / resolution).astype(int)
                time2beat[l:r+1] = 1.0

            # beat probabilities at onsets
            beat_probs = np.zeros(len(note_sequence), dtype=np.float32)
            for i in range(len(note_sequence)):
                onset = note_sequence[i][1]
                beat_probs[i] = time2beat[np.round(onset / resolution).astype(int)]

            return beat_probs

        x_beat = get_x_beat_from_ground_truth()
        
        # Get model output data
        # onset positions
        onsets = np.round(annotations['onsets_musical'] * N_per_beat)
        onsets[onsets == N_per_beat] = 0 # reset one beat onset to 0

        # note values
        note_values = np.round(annotations['note_value'] * N_per_beat)
        note_values[note_values > max_note_value] = 0 # clip note values to [0, max_note_value], index 0 will be ignored during training

        # padding
        length = len(note_sequence)
        if self.split != 'test':
            # Padding for training and validation
            max_length = training_configs['quantisation']['max_length']
            if length < max_length:
                note_sequence = np.concatenate([note_sequence, np.zeros((max_length - length, 4))])
                x_beat = np.concatenate([x_beat, np.zeros(max_length - length)])
                onsets = np.concatenate([onsets, np.zeros(max_length - length)])
                note_values = np.concatenate([note_values,  np.zeros(max_length - length)])

        return note_sequence, x_beat, onsets, note_values, length
