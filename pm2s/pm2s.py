import numpy as np
import pandas as pd

from pm2s.io.midi_read import read_note_sequence
from pm2s.features.beat import RNNJointBeatProcessor
from pm2s.features.quantisation import RNNJointQuantisationProcessor
from pm2s.features.hand_part import RNNHandPartProcessor
from pm2s.features.key_signature import RNNKeySignatureProcessor
from pm2s.features.time_signature import CNNTimeSignatureProcessor
from pm2s.constants import ticks_per_beat, resolution
from pm2s.io.midi_write import write_midi_score

def crnn_joint_pm2s(performance_midi_file, score_midi_file, start_time=0., end_time=30., include_time_signature=False, include_key_signature=True):
    """Convert a performance MIDI file into a score MIDI file using CRNN-Joint model.
    NOTE: We use the beat predictions to quantise the performance since it tends to be more accurate than the quantisation predictions.
    """
    #######################################################
    # Extract features
    #######################################################
    # Read MIDI file into note sequence
    note_seq = read_note_sequence(performance_midi_file, start_time, end_time)

    # Get the features
    beat_processor = RNNJointBeatProcessor()
    hand_part_processor = RNNHandPartProcessor()
    key_signature_processor = RNNKeySignatureProcessor()
    time_signature_processor = CNNTimeSignatureProcessor()

    beats, downbeats = beat_processor.process_note_seq(note_seq)
    hand_parts = hand_part_processor.process_note_seq(note_seq)
    key_signature_changes = key_signature_processor.process_note_seq(note_seq)
    time_signature = time_signature_processor.process_note_seq(note_seq)

    # Update the time signature based on the predicted beats and downbeats
    # Assuming no time signature changes over the segment, selecting from the basic ones among 2/4, 3/4, 4/4, 6/8
    if time_signature == '3/4':
        if len(beats) / len(downbeats) > 4.5:
            time_signature = '6/8'
        else:
            time_signature = '3/4'
    elif time_signature == '4/4':
        if len(beats) / len(downbeats) > 3:
            time_signature = '4/4'
        else:
            time_signature = '2/4'

    # Update the downbeat positions based on the predicted beats, downbeats, and time signature
    similarities = []   # similarity between the rule-based downbeat positions and the predicted downbeat positions, indexed by starting the downbeat from the idx-th beat
    for idx in range(int(time_signature.split('/')[0])):
        downbeats_rule = beats[idx::int(time_signature.split('/')[0])]
        # overlapping between downbeats_rule and downbeats
        overlap = [i for i in downbeats_rule if i in downbeats]
        similarities.append(len(overlap) / len(downbeats))
    # find the best idx
    idx = np.argmax(similarities)
    downbeats = beats[idx::int(time_signature.split('/')[0])]

    #######################################################
    # Generate MIDI score
    #######################################################
    # Get note sequence properties
    pitches = note_seq[:,0]
    onsets = note_seq[:,1]
    durations = note_seq[:,2]
    velocities = note_seq[:,3]
    length = note_seq.shape[0]
    
    # Get time2tick from beats, in mido, every beat is divided into ticks
    time2tick_dict = np.zeros(int(round(end_time / resolution)+1), dtype=int)
    
    for i in range(1, len(beats)):
        
        timeframe_left = int(round(beats[i-1] / resolution))
        timeframe_right = int(round(beats[i] / resolution))
        tick_left = (i-1) * ticks_per_beat
        tick_right = i * ticks_per_beat
        
        time2tick_dict[timeframe_left:timeframe_right+1] = np.round(np.linspace(tick_left, tick_right, timeframe_right - timeframe_left + 1)).astype(int)

    # Fill in the left and right boundaries
    # Left boundary
    if onsets[0] < beats[0]:
        # Get tick for left boundary
        timeframe_left = int(round(onsets[0] / resolution))
        timeframe_1stbeat = int(round(beats[0] / resolution))
        timeframe_2ndbeat = int(round(beats[1] / resolution))

        tick_left = -1 * ticks_per_beat * (timeframe_1stbeat - timeframe_left) / (timeframe_2ndbeat - timeframe_1stbeat)
        
        # Fill in the left boundary
        time2tick_dict[timeframe_left:timeframe_1stbeat+1] = np.round(np.linspace(tick_left, 0, timeframe_1stbeat - timeframe_left + 1)).astype(int)

        # Move tick value upwards globally (integer multiple of ticks_per_beat) so that all ticks are positive
        tick_shift_amount = int(np.ceil((0 - tick_left) / ticks_per_beat) * ticks_per_beat)
        time2tick_dict += tick_shift_amount

    # Right boundary
    if end_time > beats[-1]:
        # Get tick for right boundary
        timeframe_right = int(round(end_time / resolution))
        timeframe_lastbeat = int(round(beats[-1] / resolution))
        timeframe_2ndlastbeat = int(round(beats[-2] / resolution))

        tick_lastbeat = time2tick_dict[timeframe_lastbeat]
        tick_2ndlastbeat = time2tick_dict[timeframe_2ndlastbeat]

        tick_right = tick_lastbeat + (tick_lastbeat - tick_2ndlastbeat) * (timeframe_right - timeframe_lastbeat) / (timeframe_lastbeat - timeframe_2ndlastbeat)

        # Fill in the right boundary
        time2tick_dict[timeframe_lastbeat:timeframe_right+1] = np.round(np.linspace(tick_lastbeat, tick_right, timeframe_right - timeframe_lastbeat + 1)).astype(int)

    def time2tick(time):
        return time2tick_dict[int(round(time / resolution))]

    # Get note sequence in a DataFrame
    note_sequence = pd.DataFrame(columns=['onset_tick', 'offset_tick', 'pitch', 'velocity', 'channel']).astype(int)

    # Add notes in different tracks
    for idx in range(length):

        # Get musical onset and offset by rounding to the nearest resolution
        onset_tick = time2tick(onsets[idx])
        offset_tick = time2tick(onsets[idx] + durations[idx])
        # Round to a minimum of 48 bins per beat (i.e. 12 bins per quarter note)
        onset_tick = int(round(onset_tick / (ticks_per_beat / 48)) * (ticks_per_beat / 48))
        offset_tick = int(round(offset_tick / (ticks_per_beat / 48)) * (ticks_per_beat / 48))

        note_sequence.loc[idx] = [onset_tick, offset_tick, int(pitches[idx]), int(velocities[idx]), hand_parts[idx]]
        
    # Get tempo changes at every beat
    tempo_changes = []
    for idx in range(1, len(beats)):
        tempo = 1e+6 * (beats[idx] - beats[idx-1]) / ((time2tick(beats[idx]) - time2tick(beats[idx-1])) / ticks_per_beat)
        tempo_changes.append((time2tick(beats[idx-1]), int(tempo)))

    # Get key signature changes
    key_changes = []
    if include_key_signature:
        for ks in key_signature_changes:
            key_changes.append((time2tick(ks[0]), ks[1]))

    # Get time signature changes
    if include_time_signature:
        time_sig_changes = [(time2tick(downbeats[0]), time_signature)]
    else:
        time_sig_changes = []

    # Write MIDI file
    write_midi_score(
        note_sequence, 
        tempo_changes=tempo_changes, 
        key_signature_changes=key_changes, 
        time_signature_changes=time_sig_changes, 
        midi_file_path=score_midi_file
    )