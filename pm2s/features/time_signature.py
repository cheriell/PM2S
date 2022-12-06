import torch
import numpy as np

from pm2s.features._processor import MIDIProcessor
from pm2s.models.time_signature import RNNTimeSignatureModel
from pm2s.io.midi_read import read_note_sequence

class RNNTimeSignatureProcessor(MIDIProcessor):

    def __init__(self, model_checkpoint=None, **kwargs):
        super().__init__(model_checkpoint, **kwargs)

    def load_from_checkpoint(self, path):
        if path:
            self._model = RNNTimeSignatureModel.load_from_checkpoint(path)
        else:
            self._model = RNNTimeSignatureModel()

    def process(self, midi_file, **kwargs):
        # Read MIDI file into note sequence
        note_seq = read_note_sequence(midi_file)
        x = torch.tensor(note_seq).unsqueeze(0)

        # Forward pass
        td_probs, tn_probs = self._model(x)

        # Post-processing
        td_idx = td_probs[0].topk(1, dim=1)[1].squeeze(1).cpu().detach().numpy() # (seq_len,)
        tn_idx = tn_probs[0].topk(1, dim=1)[1].squeeze(1).cpu().detach().numpy() # (seq_len,)

        onsets = note_seq[:, 1]
        time_signature_changes = self.pps(td_idx, tn_idx, onsets)

        return time_signature_changes

    def pps(self, td_idx, tn_idx, onsets):
        ts_prev = '0/0'
        ts_changes = []
        for note_idx in range(len(td_idx)):
            ts_cur = '{:d}/{:d}'.format(tn_idx[note_idx], td_idx[note_idx])
            if note_idx == 0 or ts_cur != ts_prev:
                onset_cur = onsets[note_idx]
                ts_changes.append((onset_cur, ts_cur))
                ts_prev = ts_cur
        return ts_changes
