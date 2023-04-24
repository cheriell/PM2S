import torch
import numpy as np

from pm2s.features._processor import MIDIProcessor
from pm2s.models.quantisation import RNNJointQuantisationModel
from pm2s.io.midi_read import read_note_sequence
from pm2s.features.beat import RNNJointBeatProcessor
from pm2s.constants import N_per_beat, tolerance


class RNNJointQuantisationProcessor(MIDIProcessor):

    def __init__(self, model_state_dict_path='_model_state_dicts/quantisation/RNNJointQuantisationModel.pth', **kwargs):
        super().__init__(model_state_dict_path, **kwargs)

    def load(self, state_dict_path):
        if state_dict_path:
            self._model = RNNJointQuantisationModel()
            self._model.load_state_dict(torch.load(state_dict_path))
            self._model.eval()
        else:
            self._model = RNNJointQuantisationModel()

    def process(self, midi_file, **kwargs):
        # Read MIDI file into note sequence
        note_seq = read_note_sequence(midi_file)
        x = torch.tensor(note_seq).unsqueeze(0)

        # Forward pass
        beat_probs, downbeat_probs, onset_position_probs, note_value_probs = self._model(x)

        # Post-processing
        onset_positions_idx = onset_position_probs[0].topk(1, dim=0)[1].squeeze(0) # (n_notes,)
        note_values_idx = note_value_probs[0].topk(1, dim=0)[1].squeeze(0) # (n_notes,)

        onset_positions_idx = onset_positions_idx.detach().numpy()
        note_values_idx = note_values_idx.detach().numpy()

        beat_probs = beat_probs.squeeze(0).detach().numpy()
        downbeat_probs = downbeat_probs.squeeze(0).detach().numpy()
        onsets = note_seq[:, 1]

        beats, onset_positions, note_values = self.pps(onset_positions_idx, note_values_idx, beat_probs, downbeat_probs, onsets)

        return beats, onset_positions, note_values

    def pps(self, onset_positions_idx, note_values_idx, beat_probs, downbeat_probs, onsets):
        # post-processing
        # Convert onset position and note value indexes to actual values
        # Use predicted beat as a reference

        # get beats prediction from beat_probs and downbeat_probs
        beats = RNNJointBeatProcessor.pps(beat_probs, downbeat_probs, onsets)

        # get predicted onset positions and note values in beats
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

        return beats, onset_positions, note_values
