import numpy as np

from configs import *
from data.dataset_base import BaseDataset
from data.data_augmentation import DataAugmentation
from pm2s.constants import *


class HandPartDataset(BaseDataset):

    def __init__(self, workspace, split):
        super().__init__(workspace, split, from_asap=False)
        
        # Initialise data augmentation
        self.dataaug = DataAugmentation(feature='hand_part')

    def __getitem__(self, idx):

        row = self._sample_row(idx)
        note_sequence, annotations = self._load_data(row)

        # Get model output data
        if annotations['hands'] is None:
            hands = np.zeros(len(note_sequence), dtype=np.float64)
            hands_mask = np.zeros(len(note_sequence), dtype=np.float64)
        else:
            hands = annotations['hands'].astype(np.float64)
            hands_mask = np.ones(len(note_sequence), dtype=np.float64)

        # padding
        length = len(note_sequence)
        if length < max_length:
            note_sequence = np.concatenate([note_sequence, np.zeros((max_length - length, 4))])
            hands = np.concatenate([hands, np.zeros(max_length - length)])
            hands_mask = np.concatenate([hands_mask, np.zeros(max_length - length)])

        return note_sequence, hands, hands_mask, length