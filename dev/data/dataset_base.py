import os, sys
sys.path.insert(0, os.path.join(sys.path[0], '..'))
import torch
import pandas as pd
from collections import defaultdict
import random
import pickle
from pathlib import Path

from configs import training_configs
# from pm2s.constants import 

class BaseDataset(torch.utils.data.Dataset):

    def __init__(self, workspace, split, from_asap=True, feature=None):

        # parameters
        self.workspace = workspace
        self.feature_folder = os.path.join(workspace, 'features')
        self.split = split
        self.feature = feature

        # input checks
        assert self.split in ['train', 'valid', 'test', 'all']
        assert self.feature in ['beat', 'quantisation', 'hand_part', 'key_signature', 'time_signature']

        # Get metadata by split
        metadata = pd.read_csv('metadata/metadata.csv')
        if split == 'all':
            self.metadata = metadata
        else:
            self.metadata = metadata[metadata['split'] == split]
        if not from_asap:
            self.metadata = self.metadata[self.metadata['source'] != 'ASAP']
        self.metadata.reset_index(inplace=True)

        # Get distinct pieces
        self.piece2row = defaultdict(list)
        for i, row in self.metadata.iterrows():
            self.piece2row[row['piece_id']].append(i)
        self.pieces = list(self.piece2row.keys())

    def __len__(self):
        if self.split == 'train' or self.split == 'all':
            # constantly update 200 steps per epoch, not related to training dataset size
            return training_configs[self.feature]['batch_size'] * len(training_configs[self.feature]['gpus']) * 200

        elif self.split == 'valid':
            # by istinct pieces in validation set
            return training_configs[self.feature]['batch_size'] * len(self.piece2row)  # valid dataset size

        elif self.split == 'test':
            return len(self.metadata)

    def _sample_row(self, idx):
        # Sample one row from the metadata
        if self.split == 'train' or self.split == 'all':
            piece_id = random.choice(list(self.piece2row.keys()))   # random sampling by piece
            row_id = random.choice(self.piece2row[piece_id])
        elif self.split == 'valid':
            piece_id = self.pieces[idx // training_configs[self.feature]['batch_size']]    # by istinct pieces in validation set
            row_id = self.piece2row[piece_id][idx % training_configs[self.feature]['batch_size'] % len(self.piece2row[piece_id])]
        elif self.split == 'test':
            row_id = idx
        row = self.metadata.iloc[row_id]

        return row

    def _load_data(self, row):
        # Get feature
        note_sequence, annotations = pickle.load(open(str(Path(self.feature_folder, row['feature_file'])), 'rb'))

        # Data augmentation
        if self.split == 'train' or self.split == 'all':
            note_sequence, annotations = self.dataaug(note_sequence, annotations)

        # Randomly sample a segment that is at most max_length long
        if self.split == 'train' or self.split == 'all':
            start_idx = random.randint(0, len(note_sequence)-1)
            end_idx = start_idx + training_configs[self.feature]['max_length']
        elif self.split == 'valid':
            start_idx, end_idx = 0, training_configs[self.feature]['max_length']  # validate on the segment starting with the first note
        elif self.split == 'test':
            start_idx, end_idx = 0, len(note_sequence)  # test on the whole note sequence

        if end_idx > len(note_sequence):
            end_idx = len(note_sequence)

        note_sequence = note_sequence[start_idx:end_idx]
        for key in annotations.keys():
            if key in ['onsets_musical', 'note_value', 'hands', 'hands_mask'] and annotations[key] is not None:
                annotations[key] = annotations[key][start_idx:end_idx]

        return note_sequence, annotations
