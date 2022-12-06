import torch

from data.dataset_base import BaseDataset
from pm2s.constants import *
from configs import *

class QuantisationDataset(BaseDataset):

    def __init__(self, workspace, split):
        super().__init__(workspace, split, from_asap=False)

    def __getitem__(self, idx):

        row = self._sample_row(idx)
        note_sequence, annotations = self._load_data(row)

        # Get model output data
        # onset positions
        onsets = torch.round(annotations['onsets_musical'] * N_per_beat)
        onsets[onsets == N_per_beat] = 0 # reset one beat onset to 0

        # note values
        note_values = torch.round(annotations['note_value'] * N_per_beat)
        note_values[note_values > max_note_value] = 0 # clip note values to [0, max_note_value], index 0 will be ignored during training

        # padding
        length = len(note_sequence)
        onsets = torch.cat([onsets, torch.zeros(max_length - length, dtype=torch.float32)])
        note_values = torch.cat([note_values, torch.zeros(max_length - length, dtype=torch.float32)])

        return note_sequence, onsets, note_values, length
