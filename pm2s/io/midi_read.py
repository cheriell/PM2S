import pretty_midi
import numpy as np

def read_note_sequence(midi_file, start_time=0., end_time=None):
    """
    Load MIDI file into note sequence.

    Parameters
    ----------
    midi_file : str
        Path to MIDI file.

    Returns
    -------
    note_seq: (numpy.array) in the shape of (n, 4), where n is the number of notes, and 4 is the number of features including (pitch, onset, offset, velocity). The note sequence is sorted by onset time.
    """
    midi_data = pretty_midi.PrettyMIDI(midi_file)

    notes = []
    for instrument in midi_data.instruments:
        notes.extend(instrument.notes)
    notes = sorted(notes, key=lambda x: x.start)
    note_seq = np.array([[note.pitch, note.start, note.end - note.start, note.velocity] for note in notes])
    
    # Filter out notes that are out of the specified time range
    if end_time is None:
        end_time = midi_data.get_end_time() + 1
    note_seq = note_seq[note_seq[:, 1] >= start_time, :]
    note_seq = note_seq[note_seq[:, 1] + note_seq[:, 2] < end_time, :]
    
    return note_seq