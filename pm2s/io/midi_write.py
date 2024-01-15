import mido
from functools import cmp_to_key

from pm2s.constants import ticks_per_beat



def write_midi_score(note_sequence, tempo_changes, key_signature_changes, time_signature_changes, midi_file_path):
    """Write a MIDI file from a note sequence and other information.

    Args:
        note_sequence: pd.DataFrame
            columns: ['onset_tick', 'offset_tick', 'pitch', 'velocity', 'channel']
        tempo_changes: List[(time in tick, tempo in int)]
            e.g. [(0, 500000),...]
        key_signature_changes: List[(time in tick, key_signature in string)] 
            e.g. [(0, 'C'),...)]
        time_signature_changes: List[(time in tick, time_signature in string)] 
            e.g. [(0, '4/4'),...]
        midi_file_path: str
            Path to save the MIDI file, e.g. 'test.mid'
    """
    # Create MIDI file
    mido_data = mido.MidiFile(ticks_per_beat=ticks_per_beat)

    # Create track_0 for time, key and tempo information
    track_0 = mido.MidiTrack()
    # time signature
    for ts in time_signature_changes:
        track_0.append(
            mido.MetaMessage(
                'time_signature', 
                time=ts[0],
                numerator=int(ts[1].split('/')[0]), 
                denominator=int(ts[1].split('/')[1]),
            )
        )
    # key signature
    for ks in key_signature_changes:
        track_0.append(
            mido.MetaMessage('key_signature', time=ks[0], key=ks[1])
        )
    # tempo
    for tc in tempo_changes:
        track_0.append(
            mido.MetaMessage('set_tempo', time=tc[0], tempo=tc[1])
        )

    # Add hand parts in different tracks
    track_left = mido.MidiTrack()
    track_right = mido.MidiTrack()
    # program number (instrument: piano)
    track_left.append(mido.Message('program_change', time=0, program=0, channel=0))
    track_right.append(mido.Message('program_change', time=0, program=0, channel=1))

    # Add notes in different tracks
    for _, note in note_sequence.iterrows():

        # Get corresponding hand track
        track = track_left if note['channel'] == 0 else track_right

        # Add note
        track.append(
            mido.Message(
                'note_on',
                time=note['onset_tick'],
                note=note['pitch'],
                velocity=note['velocity'],
                channel=note['channel'],
            )
        )
        track.append(
            mido.Message(
                'note_off',
                time=note['offset_tick'],
                note=note['pitch'],
                velocity=0,
                channel=note['channel'],
            )
        )

    # Sort the events in each track by their time
    for track in [track_0, track_left, track_right]:
        track.sort(key=cmp_to_key(event_compare))
        track.append(mido.MetaMessage('end_of_track', time=track[-1].time+1))
        mido_data.tracks.append(track)

    # Convert ticks from absolute to relative
    for track in mido_data.tracks:
        tick = 0
        for event in track:
            event.time -= tick
            tick += event.time
        
    # Save MIDI file
    mido_data.save(filename=midi_file_path)


def event_compare(event1, event2):
    secondary_sort = {
        'set_tempo': lambda e: (1 * 256 * 256),
        'time_signature': lambda e: (2 * 256 * 256),
        'key_signature': lambda e: (3 * 256 * 256),
        'lyrics': lambda e: (4 * 256 * 256),
        'text_events' :lambda e: (5 * 256 * 256),
        'program_change': lambda e: (6 * 256 * 256),
        'pitchwheel': lambda e: ((7 * 256 * 256) + e.pitch),
        'control_change': lambda e: (
            (8 * 256 * 256) + (e.control * 256) + e.value),
        'note_off': lambda e: ((9 * 256 * 256) + (e.note * 256)),
        'note_on': lambda e: (
            (10 * 256 * 256) + (e.note * 256) + e.velocity) if e.velocity > 0 
            else ((9 * 256 * 256) + (e.note * 256)),
        'end_of_track': lambda e: (11 * 256 * 256)
    }
    if event1.time == event2.time and event1.type in secondary_sort and event2.type in secondary_sort:
        return secondary_sort[event1.type](event1) - secondary_sort[event2.type](event2)
    return event1.time - event2.time