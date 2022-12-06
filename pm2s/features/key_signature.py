import torch
import numpy as np

from pm2s.features._processor import MIDIProcessor
from pm2s.models.key_signature import RNNKeySignatureModel
from pm2s.io.midi_read import read_note_sequence
from pm2s.constants import keyNumber2Name

class RNNKeySignatureProcessor(MIDIProcessor):

    def __init__(self, model_checkpoint=None, **kwargs):
        super().__init__(model_checkpoint, **kwargs)

    def load_from_checkpoint(self, path):
        if path:
            self._model = RNNKeySignatureModel.load_from_checkpoint(path)
        else:
            self._model = RNNKeySignatureModel()

    def process(self, midi_file, **kwargs):
        # Read MIDI file into note sequence
        note_seq = read_note_sequence(midi_file)
        x = torch.tensor(note_seq).unsqueeze(0)

        # Forward pass
        key_probs = self._model(x)

        # Post-processing
        key_idx = key_probs[0].topk(1, dim=1)[1].squeeze(1).cpu().detach().numpy() # (seq_len,)

        onsets = note_seq[:, 1]
        key_signature_changes = self.pps(key_idx, onsets)

        return key_signature_changes

    def pps(self, key_idx, onsets):
        ks_prev = '0'
        ks_changes = []
        for note_idx in range(len(key_idx)):
            ks_cur = keyNumber2Name[key_idx[note_idx]]
            if note_idx == 0 or ks_cur != ks_prev:
                onset_cur = onsets[note_idx]
                ks_changes.append((onset_cur, ks_cur))
                ks_prev = ks_cur
        return ks_changes
            
