

class MIDIProcessor(object):
    """
    Abstract base class for processing MIDI data.
    """
    
    def __init__(self, model_state_dict_path=None, **kwargs):
        """
        Parameters
        ----------
        kwargs : dict
        """
        self._kwargs = kwargs

        self.load(model_state_dict_path)
        self._model.eval()

    def load(self, model_state_dict_path):
        """
        Load the processor from a model checkpoint file.

        Parameters
        ----------
        path : str
            Path to dump file.
        """
        raise NotImplementedError("Subclasses must implement this method.")
        
    def process(self, midi_file, **kwargs):
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

    