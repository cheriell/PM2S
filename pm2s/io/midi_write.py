import mido
from functools import cmp_to_key


def write_midi_score(mido_messages, score_midi_file, ticks_per_beat=480, voices=2):
    
    # Create MIDI file and tracks
    mido_data = mido.MidiFile(ticks_per_beat = ticks_per_beat, type=1)
    tracks = [mido.MidiTrack() for _ in range(voices+1)]  # first track for time, key and tempo changes

    # Tempo changes
    for msg in mido_messages['tempo_changes']:
        tracks[0].append(mido.MetaMessage('set_tempo', time=msg['tick'], tempo=msg['tempo']))

    # Notes
    for msg in mido_messages['notes']:
        tracks[msg['channel'] + 1].append(mido.Message('note_on', time=msg['onset_tick'], note=msg['pitch'], velocity=msg['velocity'], channel=0))
        tracks[msg['channel'] + 1].append(mido.Message('note_on', time=msg['offset_tick'], note=msg['pitch'], velocity=0, channel=0))

    # Key signature changes
    for msg in mido_messages['key_signature_changes']:
        tracks[0].append(mido.MetaMessage('key_signature', time=msg['tick'], key=msg['key']))

    # Add tracks to MIDI file
    for track in tracks:
        # Sort tracks by time
        track.sort(key=cmp_to_key(event_compare))
        # Convert ticks from absolute to relative
        tick = 0
        for event in track:
            event.time -= tick
            tick += event.time
        # Append track to MIDI
        mido_data.tracks.append(track)

    # Save MIDI file
    mido_data.save(filename=score_midi_file)

def event_compare(event1, event2):
    secondary_sort = {
        'set_tempo': lambda e: (1 * 256 * 256),
        'time_signature': lambda e: (2 * 256 * 256),
        'key_signature': lambda e: (3 * 256 * 256),
        'lyrics': lambda e: (4 * 256 * 256),
        'text_events' :lambda e: (5 * 256 * 256),
        'program_change': lambda e: (6 * 256 * 256),
        'pitchwheel': lambda e: ((7 * 256 * 256) + e.pitch),
        'control_change': lambda e: ((8 * 256 * 256) + (e.control * 256) + e.value),
        'note_on': lambda e: (
            (9 * 256 * 256) + (e.note * 256) + e.velocity) if e.velocity > 0 
            else ((10 * 256 * 256) + (e.note * 256)),
        'note_off': lambda e: ((10 * 256 * 256) + (e.note * 256)),
        'end_of_track': lambda e: (11 * 256 * 256)
    }
    if event1.time == event2.time and event1.type in secondary_sort and event2.type in secondary_sort:
        return secondary_sort[event1.type](event1) - secondary_sort[event2.type](event2)
    return event1.time - event2.time