import torch
import numpy as np

from pm2s.features._processor import MIDIProcessor
from pm2s.models.time_signature import RNNTimeSignatureModel
from pm2s.io.midi_read import read_note_sequence
from pm2s.constants import tsIndex2Nume, tsIndex2Deno

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
        td_probs, tn_probs = self._model(x)

        # Post-processing
        td_idx = td_probs[0].topk(1, dim=0)[1].squeeze(0).cpu().detach().numpy() # (seq_len,)
        tn_idx = tn_probs[0].topk(1, dim=0)[1].squeeze(0).cpu().detach().numpy() # (seq_len,)

        onsets = note_seq[:, 1]
        time_signature_changes = self.pps(td_idx, tn_idx, onsets)

        return time_signature_changes

    def pps(self, td_idx, tn_idx, onsets):
        ts_prev = '0/0'
        ts_changes = []
        for i in range(len(td_idx)):
            ts_cur = '{:d}/{:d}'.format(tsIndex2Nume[tn_idx[i]], tsIndex2Deno[td_idx[i]])
            if i == 0 or ts_cur != ts_prev:
                onset_cur = onsets[i]
                ts_changes.append((onset_cur, ts_cur))
                ts_prev = ts_cur
        return ts_changes
