import torch

from pm2s.features._processor import MIDIProcessor
from pm2s.models.hand_part import RNNHandPartModel
from pm2s.io.midi_read import read_note_sequence
from pm2s.constants import model_state_dict_paths


class RNNHandPartProcessor(MIDIProcessor):

    def __init__(self, **kwargs):
        model_state_dict_path = model_state_dict_paths['hand_part']['state_dict_path']
        super().__init__(model_state_dict_path, **kwargs)

    def load(self, state_dict_path):
        if state_dict_path:
            self._model = RNNHandPartModel()
            self._model.load_state_dict(torch.load(state_dict_path))
        else:
            self._model = RNNHandPartModel()

    def process(self, midi_file, **kwargs):
        # Read MIDI file into note sequence
        note_seq = read_note_sequence(midi_file)
        x = torch.tensor(note_seq).unsqueeze(0)

        # Forward pass
        hand_probs = self._model(x)

        # Post-processing
        hand_probs = hand_probs.squeeze(0).cpu().detach().numpy()
        hand_parts = (hand_probs > 0.5).astype(int)

        return hand_parts
    