import torch

from pm2s.features._processor import MIDIProcessor
from pm2s.models.hand_part import RNNHandPartModel
from pm2s.io.midi_read import read_note_sequence


class RNNHandPartProcessor(MIDIProcessor):

    def __init__(self, model_checkpoint=None, **kwargs):
        super().__init__(model_checkpoint, **kwargs)

    def load_from_checkpoint(self, path):
        if path:
            self._model = RNNHandPartModel.load_from_checkpoint(path)
        else:
            self._model = RNNHandPartModel()

    def process(self, midi_file, **kwargs):
        # Read MIDI file into note sequence
        note_seq = read_note_sequence(midi_file)
        x = torch.tensor(note_seq).unsqueeze(0)

        # Forward pass
        hand_probs = self._model(x)

        # Post-processing
        hand_probs = hand_probs.squeeze(2).squeeze(0).cpu().detach().numpy()

        hand_probs = hand_probs > 0.5

        return hand_probs
    