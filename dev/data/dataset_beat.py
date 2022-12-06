import torch
import numpy as np

from configs import *
from data.dataset_base import BaseDataset
from pm2s.constants import *


class BeatDataset(BaseDataset):

    def __init__(self, workspace, split):
        super().__init__(workspace, split)

    def __getitem__(self, idx):

        row = self._sample_row(idx)
        note_sequence, annotations = self._load_data(row)

        # Get model output data
        # beats downbeats
        beats = annotations['beats']
        downbeats = annotations['downbeats']

        # time to beat/downbeat/inter-beat-interval dictionaries
        end_time = max(beats[-1], note_sequence[-1][1] + note_sequence[-1][2]) + 1.0
        time2beat = np.zeros(int(np.ceil(end_time / resolution)))
        time2downbeat = np.zeros(int(np.ceil(end_time / resolution)))
        time2ibi = np.zeros(int(np.ceil(end_time / resolution)))
        for idx, beat in enumerate(beats):
            l = np.round((beat - tolerance) / resolution).astype(int)
            r = np.round((beat + tolerance) / resolution).astype(int)
            time2beat[l:r+1] = 1.0

            ibi = beats[idx+1] - beats[idx] if idx+1 < len(beats) else beats[-1] - beats[-2]
            l = np.round((beat - tolerance) / resolution).astype(int) if idx > 0 else 0
            r = np.round((beat + ibi) / resolution).astype(int) if idx+1 < len(beats) else len(time2ibi)
            if ibi > 4:
                # reset ibi to 0 if it's too long, index 0 will be ignored during training
                ibi = np.array(0)
            time2ibi[l:r+1] = np.round(ibi / resolution)
        
        for downbeat in downbeats:
            l = np.round((downbeat - tolerance) / resolution).astype(int)
            r = np.round((downbeat + tolerance) / resolution).astype(int)
            time2downbeat[l:r+1] = 1.0
        
        # get beat probabilities at note onsets
        beat_probs = np.zeros(len(note_sequence), dtype=np.float32)
        downbeat_probs = np.zeros(len(note_sequence), dtype=np.float32)
        ibis = np.zeros(len(note_sequence), dtype=np.float32)
        for i in range(len(note_sequence)):
            onset = note_sequence[i][1]
            beat_probs[i] = time2beat[np.round(onset / resolution).astype(int)]
            downbeat_probs[i] = time2downbeat[np.round(onset / resolution).astype(int)]
            ibis[i] = time2ibi[np.round(onset / resolution).astype(int)]
        
        # pad if length is shorter than max_length
        length = len(note_sequence)
        if len(note_sequence) < max_length:
            note_sequence = np.concatenate([note_sequence, np.zeros((max_length - len(note_sequence), 4))])
            beat_probs = np.concatenate([beat_probs, np.zeros(max_length - len(beat_probs))])
            downbeat_probs = np.concatenate([downbeat_probs, np.zeros(max_length - len(downbeat_probs))])
            ibis = np.concatenate([ibis, np.zeros(max_length - len(ibis))])
            
        return (
            note_sequence, 
            beat_probs, 
            downbeat_probs, 
            ibis, 
            length,
        )
