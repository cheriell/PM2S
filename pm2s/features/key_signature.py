import torch
import numpy as np

from pm2s.features._processor import MIDIProcessor
from pm2s.models.key_signature import RNNKeySignatureModel
from pm2s.io.midi_read import read_note_sequence
from pm2s.constants import keyNumber2Name, model_state_dict_paths

class RNNKeySignatureProcessor(MIDIProcessor):

    def __init__(self, state_dict_path=None):
        if state_dict_path is None:
            state_dict_path = model_state_dict_paths['key_signature']['state_dict_path']
        zenodo_path = model_state_dict_paths['key_signature']['zenodo_path']

        self._model = RNNKeySignatureModel()
        self.load(state_dict_path=state_dict_path, zenodo_path=zenodo_path)

    def process_note_seq(self, note_seq):
        # Process note sequence

        x = torch.tensor(note_seq).unsqueeze(0)

        # Forward pass
        key_probs = self._model(x)

        # Post-processing
        key_idx = key_probs[0].topk(1, dim=0)[1].squeeze(0).cpu().detach().numpy() # (seq_len,)

        onsets = note_seq[:, 1]
        key_signature_changes = self.pps(key_idx, onsets)

        return key_signature_changes

    def pps(self, key_idx, onsets):
        ks_prev = '0'
        ks_changes = []
        for i in range(len(key_idx)):
            ks_cur = keyNumber2Name[key_idx[i]]
            if i == 0 or ks_cur != ks_prev:
                onset_cur = onsets[i]
                ks_changes.append((onset_cur, ks_cur))
                ks_prev = ks_cur
        return ks_changes
            
