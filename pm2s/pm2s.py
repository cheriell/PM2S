import numpy as np
import pandas as pd

from pm2s.io.midi_read import read_note_sequence
from pm2s.features.beat import RNNJointBeatProcessor
from pm2s.features.quantisation import RNNJointQuantisationProcessor
from pm2s.features.hand_part import RNNHandPartProcessor
from pm2s.features.key_signature import RNNKeySignatureProcessor
from pm2s.features.time_signature import CNNTimeSignatureProcessor
from pm2s.io.midi_write import write_midi_score
from pm2s.io.utilities import (
    sec2microsec,
    microsec2sec, 
    get_possible_tick_remainders, 
    round_tick_remainder,
)

class CRNNJointPM2S:
    """CRNN-Joint model for performance MIDI to score MIDI conversion."""

    def __init__(self,
        model_path_beat : str = None,
        model_path_hand_part : str = None,
        model_path_key_sig : str = None,
        model_path_time_sig : str = None,
        beat_pps_args: dict = {
            'prob_thresh': 0.5,
            'penalty': 1.0,
            'merge_downbeats': True,
            'method': 'dp',
        },
        ticks_per_beat : int = 480,
        notes_per_beat : list = [1, 2, 3, 4, 6, 8],
    ):
        """Initialise the CRNN-Joint model for performance MIDI to score MIDI conversion.
        
        Args:
            model_path_beat: str
                Path to the beat model state dict.
            model_path_hand_part: str
                Path to the hand part model state dict.
            model_path_key_sig: str
                Path to the key signature model state dict.
            model_path_time_sig: str
                Path to the time signature model state dict.
            ticks_per_beat: int
                Number of MIDI ticks per quarter note.
            notes_per_beat: list
                Possible number of notes (in terms of note duration) per quarter note duration.
        """
        
        # Processors
        self.beat_processor = RNNJointBeatProcessor(state_dict_path=model_path_beat)
        self.hand_part_processor = RNNHandPartProcessor(state_dict_path=model_path_hand_part)
        self.key_signature_processor = RNNKeySignatureProcessor(state_dict_path=model_path_key_sig)
        self.time_signature_processor = CNNTimeSignatureProcessor(state_dict_path=model_path_time_sig)

        self.beat_pps_args = beat_pps_args
        
        self.ticks_per_beat = ticks_per_beat
        self.notes_per_beat = notes_per_beat

    def convert(self,
        performance_midi_file : str,
        score_midi_file : str,
        start_time : float = 0.,
        end_time : float = None,
        include_time_signature : bool = False,
        include_key_signature : bool = True,
    ):
        """Convert a performance MIDI file into a score MIDI file using CRNN-Joint model.
        
        Args:
            performance_midi_file: str
                Path to the performance MIDI file.
            score_midi_file: str
                Path to the score MIDI file.
            start_time: float
                Start time of the segment to be converted.
            end_time: float
                End time of the segment to be converted. If None, set to the duration of the performance MIDI.
            include_time_signature: bool
                Include time signature in the score MIDI file.
            include_key_signature: bool
                Include key signature in the score MIDI file.
        """

        # Read MIDI file into note sequence
        print('getting note sequence')
        note_seq = read_note_sequence(performance_midi_file, start_time, end_time)

        # Get score features
        print('getting score features')
        (
            beats, downbeats,
            hand_parts, 
            key_signature_changes,
            time_signature,
        ) = self.get_score_features(note_seq)

        # Prepare time to tick conversion
        print('preparing time to tick')
        time_tick_dict = self.prepare_time2tick(
            beats = beats,
            note_times = np.concatenate([note_seq[:, 1], note_seq[:, 1] + note_seq[:, 2]]),
            start_time = start_time,
            end_time = end_time,
        )

        # Get mido messages
        print('getting mido messages')
        mido_messages = self.get_mido_messages(
            note_seq = note_seq,
            hand_parts = hand_parts,
            key_signature_changes = key_signature_changes,
            time_signature_changes = [time_signature],
            time_tick_dict = time_tick_dict,
            include_time_signature = include_time_signature,
            include_key_signature = include_key_signature,
        )

        # Write the score MIDI file
        print('writing midi score')
        write_midi_score(mido_messages, score_midi_file, self.ticks_per_beat)
        
    def get_score_features(self, note_seq):

        beats, downbeats = self.beat_processor.process_note_seq(note_seq, pps_args=self.beat_pps_args)
        hand_parts = self.hand_part_processor.process_note_seq(note_seq)
        key_signature_changes = self.key_signature_processor.process_note_seq(note_seq)
        time_signature = self.time_signature_processor.process_note_seq(note_seq)

        # Update the time signature based on the predicted beats and downbeats
        # Assuming no time signature changes over the segment, selecting from the basic ones among 2/4, 3/4, 4/4, 6/8
        if time_signature == '3/4':
            if len(beats) / len(downbeats) > 4.5:
                time_signature = '6/8'
        elif time_signature == '4/4':
            if len(beats) / len(downbeats) < 3:
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

        return (
            beats, downbeats,
            hand_parts, 
            key_signature_changes,
            time_signature,
        )
    
    def prepare_time2tick(self,
            beats, 
            note_times, 
            start_time, 
            end_time, 
        ):
        """Time to tick conversion"""

        # Initialse a dict for (time: tick), where time is in microseconds (int)
        time_tick_dict = {}

        ###########################################
        # Beat-level time-tick pairs
        ###########################################

        # Track beat index (predicted beats, and other unpredicted beats at the beginning and end of the segment)
        beat_idx = 0

        # Add start time at tick 0
        if beats[0] > start_time:
            # First beat tick at start_time
            time_tick_dict[sec2microsec(start_time)] = 0
            beat_idx += 1

            # Add potential beats between the start time and the first beat
            ibi =  beats[1] - beats[0]
            if beats[0] - start_time > ibi:

                tm_list = []
                tm = beats[0] - ibi
                while tm > start_time:
                    tm_list.append(tm)
                    tm -= ibi
                tm_list = tm_list[::-1]

                for tm in tm_list:
                    time_tick_dict[sec2microsec(tm)] = beat_idx * self.ticks_per_beat
                    beat_idx += 1
            
        # Add beats at integer multiples of ticks_per_beat
        for i in range(len(beats)):
            if beats[i] > start_time and beats[i] < end_time:
                time_tick_dict[sec2microsec(beats[i])] =  beat_idx * self.ticks_per_beat
                beat_idx += 1

        # Add end time at the last beat
        if beats[-1] < end_time:
            # Possible beats between the last predicted beat and the end_time
            ibi = beats[-1] - beats[-2]
            if end_time - beats[-1] > ibi:

                tm_list = []
                tm = beats[-1] + ibi
                while tm < end_time:
                    time_tick_dict[sec2microsec(tm)] = beat_idx * self.ticks_per_beat
                    beat_idx += 1
                    tm += ibi

            # Last beat tick at end_time
            time_tick_dict[sec2microsec(end_time)] = beat_idx * self.ticks_per_beat
            beat_idx += 1

        assert np.min(np.diff(list(time_tick_dict.keys()))) >= 0, 'Time ticks are not sorted!'
        assert np.min(np.diff(list(time_tick_dict.values()))) >= 0, 'Tick values are not sorted!'
        
        ###########################################
        # Note-level time-tick pairs
        ###########################################

        # Add note_times to the time_tick_dict list
        tick_remainders = get_possible_tick_remainders(self.ticks_per_beat, self.notes_per_beat)
        note_times = np.sort(note_times)

        ms_list = list(time_tick_dict.keys())
        ms_idx = 0
        tick_prev = 0

        for i in range(len(note_times)):
            if note_times[i] > start_time and note_times[i] < end_time:

                ms_cur = sec2microsec(note_times[i])
                if ms_cur in time_tick_dict.keys():
                    continue
                while ms_cur > ms_list[ms_idx+1]:
                    ms_idx += 1
                    
                # left < cur <= right
                ms_l = ms_list[ms_idx]
                ms_r = ms_list[ms_idx+1]
                tick_l = time_tick_dict[ms_l]

                assert ms_cur >= ms_l and ms_cur <= ms_r, 'Time value is not in the interval!, ms_l: {}, ms_cur: {}, ms_r: {}'.format(ms_l, ms_cur, ms_r)

                tick = tick_l + self.ticks_per_beat * (ms_cur - ms_l) / (ms_r - ms_l)
                # Round to notes_per_beat
                tick = round_tick_remainder(tick, tick_remainders, self.ticks_per_beat)
                
                # Add to the time_tick_dict
                time_tick_dict[ms_cur] = tick
                
                assert tick >= tick_prev, 'Tick values are not sorted!, tick: {}, tick_prev: {}'.format(tick, tick_prev)
                tick_prev = tick

        
        # sort the dict by keys
        time_tick_dict = dict(sorted(time_tick_dict.items()))
        assert np.min(np.diff(list(time_tick_dict.values()))) >= 0, 'Tick values are not sorted!\nTicks:\n{}'.format(time_tick_dict.values())

        return time_tick_dict

    def get_mido_messages(self,
        note_seq,
        hand_parts,
        key_signature_changes,
        time_signature_changes,
        time_tick_dict,
        include_time_signature = False,
        include_key_signature = True,
    ):
        """Get the mido messages for the score MIDI file. Ticks are in absolute time!"""

        # Initialise the mido messages
        mido_messages = {
            'tempo_changes' : [],
            'notes' : [],
            'key_signature_changes' : [],
            'time_signature_changes' : [],
        }

        # Tempo changes
        time_list = list(time_tick_dict.keys())
        tick_list = list(time_tick_dict.values())

        time_prev, tick_prev = 0, 0
        for i in range(len(time_tick_dict)):
            
            if tick_list[i] == tick_prev:
                continue

            time_cur = time_list[i]
            tick_cur = tick_list[i]
            tempo = self.ticks_per_beat * (time_cur - time_prev) / (tick_cur - tick_prev) # microseconds per beat
            tempo = int(np.round(tempo))

            assert tempo > 0 and tempo < 8 * 1e+6, 'Tempo is out of range!, tempo: {}, time_prev: {}, time_cur: {}, beat_prev: {}, beat_cur: {}'.format(tempo, microsec2sec(time_prev), microsec2sec(time_cur), tick_prev / self.ticks_per_beat, tick_cur / self.ticks_per_beat)

            mido_messages['tempo_changes'].append({
                'tick': tick_prev, 
                'tempo': tempo
            })
            time_prev = time_cur
            tick_prev = tick_cur

        # Notes
        for i in range(len(note_seq)):
            onset_tick = time_tick_dict[sec2microsec(note_seq[i, 1])]
            offset_tick = time_tick_dict[sec2microsec(note_seq[i, 1] + note_seq[i, 2])]
            mido_messages['notes'].append({
                'onset_tick' : onset_tick,
                'offset_tick' : offset_tick,
                'pitch' : int(note_seq[i, 0]),
                'velocity' : int(note_seq[i, 3]),
                'channel' : hand_parts[i],
            })

        # Key signature changes
        if include_key_signature:
            for ks in key_signature_changes:
                mido_messages['key_signature_changes'].append({
                    'tick': time_tick_dict[sec2microsec(ks[0])],
                    'key': ks[1]
                })

        return mido_messages

