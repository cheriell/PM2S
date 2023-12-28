import torch
import numpy as np

from pm2s.features._processor import MIDIProcessor
from pm2s.models.time_signature import RNNTimeSignatureModel
from pm2s.io.midi_read import read_note_sequence
from pm2s.constants import tsIndex2Nume, tsIndex2Deno, Index2timeSig

class RNNTimeSignatureProcessor(MIDIProcessor):

    def __init__(self, model_state_dict_path='_model_state_dicts/time_signature/RNNTimeSignatureModel.pth', **kwargs):
        super().__init__(model_state_dict_path, **kwargs)

    def load(self, state_dict_path):
        if state_dict_path:
            self._model = RNNTimeSignatureModel()
            self._model.load_state_dict(torch.load(state_dict_path))
        else:
            self._model = RNNTimeSignatureModel()

    def process(self, midi_file, **kwargs):
        # Read MIDI file into note sequence
        note_seq = read_note_sequence(midi_file)
        x = torch.tensor(note_seq).unsqueeze(0)

        # Forward pass
        ts_probs = self._model(x)

        # Post-processing
        ts_probs = ts_probs.squeeze(0).detach().numpy  # (seq_len,)
        onsets = note_seq[:, 1]
        time_signature_changes = self.pps(ts_probs, onsets)

        return time_signature_changes
    
    def pps(self, ts_probs, onsets):
        # ts_probs: (seq_len,)
        # onsets: (seq_len,)
        ts_prev = '4/4'
        ts_changes = []
        for i in range(len(ts_probs)):
            ts_cur = '4/4' if ts_probs < 0.5 else '3/4'
            if i == 0 or ts_cur != ts_prev:
                onset_cur = onsets[i]
                ts_changes.append((onset_cur, ts_cur))
                ts_prev = ts_cur
        return ts_changes
