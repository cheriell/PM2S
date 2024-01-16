import numpy as np

from configs import training_configs
from data.dataset_base import BaseDataset
from data.data_augmentation import DataAugmentation
from pm2s.constants import tolerance


class TimeSignatureDataset(BaseDataset):

    def __init__(self, workspace, split):
        super().__init__(workspace, split, feature='time_signature')
        
        # Initialise data augmentation
        self.dataaug = DataAugmentation(feature='time_signature')

    def __getitem__(self, idx):

        row = self._sample_row(idx)
        note_sequence, annotations = self._load_data(row)

        # Get model output data
        time_signatures = annotations['time_signatures']
        timeSigs = np.zeros(len(note_sequence)).astype(float)

        for i in range(len(note_sequence)):
            onset = note_sequence[i, 1]
            for ts in time_signatures:
                if ts[0] > onset + tolerance:
                    break

                # Change to use binary classification (0: 4-based time signature, 1: 3-based time signature)
                if ts[1] % 3 == 0:
                    timeSigs[i] = 1
                else:
                    timeSigs[i] = 0

        # padding
        length = len(note_sequence)
        if self.split != 'test':
            # Padding for training and validation
            max_length = training_configs['time_signature']['max_length']
            if length < max_length:
                note_sequence = np.concatenate([note_sequence, np.zeros((max_length - length, 4))])
                timeSigs = np.concatenate([timeSigs, np.zeros(max_length - length)])

        return note_sequence, timeSigs, length

