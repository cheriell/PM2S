import numpy as np

from configs import training_configs
from data.dataset_base import BaseDataset
from data.data_augmentation import DataAugmentation
from pm2s.constants import *


class KeySignatureDataset(BaseDataset):

    def __init__(self, workspace, split, mode):
        super().__init__(workspace, split, feature='key_signature', mode=mode)
        
        # Initialise data augmentation
        self.dataaug = DataAugmentation(feature='key_signature')

    def __getitem__(self, idx):

        row = self._sample_row(idx)
        note_sequence, annotations = self._load_data(row)

        # Get model output data
        key_signatures = annotations['key_signatures']

        key_numbers = np.zeros(len(note_sequence)).astype(float)

        for i in range(len(note_sequence)):
            onset = note_sequence[i,1]
            for ks in key_signatures:
                if ks[0] > onset + tolerance:
                    break
                key_numbers[i] = ks[1] % keyVocabSize

        # padding
        length = len(note_sequence)
        if self.split != 'test':
            # Padding for training and validation
            max_length = training_configs['key_signature']['max_length']
            if length < max_length:
                note_sequence = np.concatenate([note_sequence, np.zeros((max_length - length, 4))])
                key_numbers = np.concatenate([key_numbers, np.zeros(max_length - length)])

        return note_sequence, key_numbers, length