import os
import torch

from pm2s.io.midi_read import read_note_sequence

class MIDIProcessor(object):
    """
    Abstract base class for processing MIDI data.
    """
    def __init__(self, state_dict_path=None):
        """
        Initialize the processor.

        Parameters
        ----------
        state_dict_path : str
            Path to model checkpoint file.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def load(self, state_dict_path, zenodo_path):
        """Load the processor from a model checkpoint file.
        """
        if not os.path.exists(state_dict_path):
            print('Downloading model_state_dict from Zenodo...')
            if not os.path.exists(os.path.dirname(state_dict_path)):
                os.makedirs(os.path.dirname(state_dict_path))
            os.system('wget -O "{}" "{}"'.format(state_dict_path, zenodo_path))
            
        self._model.load_state_dict(torch.load(state_dict_path))

        self._model.eval()

    def process(self, midi_file, **kwargs):
        # Set model to evaluation mode
        self._model.eval()
        # Read MIDI file into note sequence
        note_seq = read_note_sequence(midi_file)

        return self.process_note_seq(note_seq)
        
    def process_note_seq(self, midi_file, **kwargs):
        """
        Process the input MIDI file.

        Parameters
        ----------
        midi_file : str
        
        Returns
        -------
        Depends on the processor.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def __call__(self, midi_file, **kwargs):
        """
        Process the input MIDI file.

        Parameters
        ----------
        midi_file : str

        Returns
        -------
        Depends on the processor.
        """
        return self.process(midi_file, **kwargs)

    