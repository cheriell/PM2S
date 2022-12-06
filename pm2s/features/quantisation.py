import torch
import numpy as np

from pm2s.features._processor import MIDIProcessor
from pm2s.models.quantisation import RNNJointQuantisationModel
from pm2s.io.midi_read import read_note_sequence
from pm2s.features.beat import RNNJointBeatProcessor
from pm2s.constants import N_per_beat


class RNNJointQuantisationProcessor(MIDIProcessor):

    def __init__(self, model_checkpoint=None, **kwargs):
        super().__init__(model_checkpoint, **kwargs)

    def load_from_checkpoint(self, path):
        if path:
            self._model = RNNJointQuantisationModel.load_state_dict(path, beat_model_checkpoint=self._kwargs['beat_model_checkpoint'])
        else:
            self._model = RNNJointQuantisationModel(self._kwargs['beat_model_checkpoint'])

    def process(self, midi_file, **kwargs):
        # Read MIDI file into note sequence
        note_seq = read_note_sequence(midi_file)
        x = torch.tensor(note_seq).unsqueeze(0)

        # Forward pass
        beat_probs, downbeat_probs, onset_position_probs, note_value_probs = self._model(x)

        # Post-processing
        onset_positions_idx = onset_position_probs[0].topk(1, dim=1)[1].squeeze(1) # (n_notes,)
        note_values_idx = note_value_probs[0].topk(1, dim=1)[1].squeeze(1) # (n_notes,)

        onset_positions_idx = onset_positions_idx.detach().numpy()
        note_values_idx = note_values_idx.detach().numpy()

        beat_probs = beat_probs.squeeze(0).detach().numpy()
        downbeat_probs = downbeat_probs.squeeze(0).detach().numpy()
        onsets = note_seq[:, 1]

        onset_positions, note_values = self.pps(onset_positions_idx, note_values_idx, beat_probs, downbeat_probs, onsets)

        return onset_positions, note_values

    def pps(self, onset_positions_idx, note_values_idx, beat_probs, downbeat_probs, onsets):
        # post-processing
        # Convert onset position and note value indexes to actual values
        # Use predicted beat as a reference

        beats = RNNJointBeatProcessor.pps_dp(beat_probs, downbeat_probs, onsets)
        # TODO: trim beats to fill the whole song

        onset_positions_raw = onset_positions_idx / N_per_beat
        note_values_raw = note_values_idx / N_per_beat

        # Convert onset positions within a beat to absolute positions
        onset_positions = np.zeros_like(onset_positions_raw)
        note_values = np.zeros_like(note_values_raw)

        beat_idx = 0
        note_idx = 0

        while note_idx < len(onset_positions_raw):
            if beat_idx < len(beats) and onsets[note_idx] >= beats[beat_idx]:
                beat_idx += 1

            onset_positions[note_idx] = onset_positions_raw[note_idx] + beat_idx
            note_values[note_idx] = note_values_raw[note_idx]
            note_idx += 1

        # TODO: trim onset positions and note values to fill the whole song

        return onset_positions, note_values
