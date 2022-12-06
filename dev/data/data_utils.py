import os, sys
sys.path.insert(0, os.path.join(sys.path[0], '..'))
import torch
import random
import pretty_midi as pm
import pandas as pd
import numpy as np
from functools import reduce, cmp_to_key
from pathlib import Path

from pm2s.constants import (
    resolution, 
    tolerance, 
    keySharps2Number, 
    keyVocabSize,
    tsDeno2Index, 
    # max_length_pr
)
    
def get_note_sequence_from_midi(midi_file):
    """
    Get note sequence from midi file.
    Note sequence is in a list of (pitch, onset, duration, velocity) tuples, in np.array.
    """
    midi_data = pm.PrettyMIDI(str(Path(midi_file)))
    note_sequence = reduce(lambda x, y: x+y, [inst.notes for inst in midi_data.instruments])
    note_sequence = sorted(note_sequence, key=cmp_to_key(compare_note_order))
    # conver to numpy array
    note_sequence = np.array([(note.pitch, note.start, note.end-note.start, note.velocity) \
                                    for note in note_sequence])
    return note_sequence

def get_annotations_from_annot_file(annot_file):
    """
    Get annotations from annotation file in ASAP dataset.
    annotatioins in a dict of {
        beats: list of beat times,
        downbeats: list of downbeat times,
        time_signatures: list of (time, numerator, denominator) tuples,
        key_signatures: list of (time, key_number) tuples
    }, all in np.array.
    """
    annot_data = pd.read_csv(str(Path(annot_file)), header=None, sep='\t')

    beats, downbeats, key_signatures, time_signatures = [], [], [], []
    for i, row in annot_data.iterrows():
        a = row[2].split(',')
        # beats
        beats.append(row[0])
        # downbeats
        if a[0] == 'db':
            downbeats.append(row[0])
        # time_signatures
        if len(a) >= 2 and a[1] != '':
            numerator, denominator = a[1].split('/')
            time_signatures.append((row[0], int(numerator), int(denominator)))
        # key_signatures
        if len(a) == 3 and a[2] != '':
            key_signatures.append((row[0], keySharps2Number[int(a[2])]))

    # save as annotation dict
    annotations = {
        'beats': np.array(beats),
        'downbeats': np.array(downbeats),
        'time_signatures': np.array(time_signatures),
        'key_signatures': np.array(key_signatures),
        'onsets_musical': None,
        'note_value': None,
        'hands': None,
    }
    return annotations

def get_note_sequence_and_annotations_from_midi(midi_file):
    """
    Get beat sequence and annotations from midi file.
    Note sequence is in a list of (pitch, onset, duration, velocity) tuples, in np.array.
    annotations in a dict of {
        beats: list of beat times,
        downbeats: list of downbeat times,
        time_signatures: list of (time, numerator, denominator) tuples,
        key_signatures: list of (time, key_number) tuples,
        onsets_musical: list of onsets in musical time for each note (within a beat),
        note_value: list of note values (in beats),
        hands: list of hand part for each note (0: left, 1: right)
    """
    midi_data = pm.PrettyMIDI(str(Path(midi_file)))

    # note sequence and hands
    if len(midi_data.instruments) == 2:
        # two hand parts
        note_sequence_with_hand = []
        for hand, inst in enumerate(midi_data.instruments):
            for note in inst.notes:
                note_sequence_with_hand.append((note, hand))

        def compare_note_with_hand(x, y):
            return compare_note_order(x[0], y[0])
        note_sequence_with_hand = sorted(note_sequence_with_hand, key=cmp_to_key(compare_note_with_hand))

        note_sequence, hands = [], []
        for note, hand in note_sequence_with_hand:
            note_sequence.append(note)
            hands.append(hand)
    else:
        # ignore data with other numbers of hand parts
        note_sequence = reduce(lambda x, y: x+y, [inst.notes for inst in midi_data.instruments])
        note_sequence = sorted(note_sequence, key=cmp_to_key(compare_note_order))
        hands = None

    # beats
    beats = midi_data.get_beats()
    # downbeats
    downbeats = midi_data.get_downbeats()
    # time_signatures
    time_signatures = [(t.time, t.numerator, t.denominator) for t in midi_data.time_signature_changes]
    # key_signatures
    key_signatures = [(k.time, k.key_number) for k in \
                        midi_data.key_signature_changes]
    # onsets_musical and note_values
    def time2pos(t):
        # convert time to position in musical time within a beat (unit: beat, range: 0-1)
        # after checking, we confirmed that beats[0] is always 0
        idx = np.where(beats - t <= tolerance)[0][-1]
        if idx+1 < len(beats):
            base = midi_data.time_to_tick(beats[idx+1]) - midi_data.time_to_tick(beats[idx])
        else:
            base = midi_data.time_to_tick(beats[-1]) - midi_data.time_to_tick(beats[-2])
        return (midi_data.time_to_tick(t) - midi_data.time_to_tick(beats[idx])) / base

    def times2note_value(start, end):
        # convert start and end times to note value (unit: beat, range: 0-4)
        idx = np.where(beats - start <= tolerance)[0][-1]
        if idx+1 < len(beats):
            base = midi_data.time_to_tick(beats[idx+1]) - midi_data.time_to_tick(beats[idx])
        else:
            base = midi_data.time_to_tick(beats[-1]) - midi_data.time_to_tick(beats[-2])
        return (midi_data.time_to_tick(end) - midi_data.time_to_tick(start)) / base

    # get onsets_musical and note_values
    # filter out small negative values (they are usually caused by errors in time_to_tick convertion)
    onsets_musical = [min(1, max(0, time2pos(note.start))) for note in note_sequence]  # in range 0-1
    note_values = [max(0, times2note_value(note.start, note.end)) for note in note_sequence]

    # conver to Tensor
    note_sequence = np.array([[note.pitch, note.start, note.end-note.start, note.velocity] \
                                    for note in note_sequence])
    # save as annotation dict
    annotations = {
        'beats': np.array(beats),
        'downbeats': np.array(downbeats),
        'time_signatures': np.array(time_signatures),
        'key_signatures': np.array(key_signatures),
        'onsets_musical': np.array(onsets_musical),
        'note_value': np.array(note_values),
        'hands': np.array(hands) if hands is not None else None,
    }
    return note_sequence, annotations

def compare_note_order(note1, note2):
    """
    Compare two notes by firstly onset and then pitch.
    """
    if note1.start < note2.start:
        return -1
    elif note1.start == note2.start:
        if note1.pitch < note2.pitch:
            return -1
        elif note1.pitch == note2.pitch:
            return 0
        else:
            return 1
    else:
        return 1

# def get_baseline_model_output_data(note_sequence, annotations, sample_segment=True):
#     """
#     Get beat and downbeat activation from beat and downbeat sequence.
#     """

#     # get valid length for the pianorolls
#     length = (torch.max(note_sequence[:,1] + note_sequence[:,2]) * (1 / resolution) + 1).long()
#     length = torch.min(length, torch.tensor(max_length_pr)).long()

#     # start time of the segment
#     t0 = note_sequence[0, 1]

#     # get beat and downbeat activation functions
#     beats = annotations['beats']
#     downbeats = annotations['downbeats']
#     beat_act = torch.zeros(max_length_pr).float()
#     downbeat_act = torch.zeros(max_length_pr).float()

#     for beat in beats:
#         left = int(min(length, max(0, torch.round((beat - t0) / resolution) - 1)))
#         right = int(min(length, max(0, torch.round((beat - t0) / resolution) + 1)))
#         beat_act[left:right+1] = 1.0
#     for downbeat in downbeats:
#         left = int(min(length, max(0, torch.round((downbeat - t0) / resolution) - 1)))
#         right = int(min(length, max(0, torch.round((downbeat - t0) / resolution) + 1)))
#         downbeat_act[left:right+1] = 1.0

#     return beat_act, downbeat_act, length