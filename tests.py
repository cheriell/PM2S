# =============================================================================

# from pm2s.features.beat import RNNJointBeatProcessor

# # Path to the MIDI recording
# midi_recording = '/import/c4dm-05/ll307/datasets/A-MAPS_1.1/MAPS_MUS-alb_esp2_SptkBGAm.mid'

# # Create a beat processor
# processor = RNNJointBeatProcessor()

# # Process the MIDI recording to the beat predictions
# beats = processor.process(midi_recording)
# print(beats)


# =============================================================================

# import numpy as np
# import pretty_midi as pm
# import matplotlib.pyplot as plt

# def get_piano_roll(midi_file, start_time, end_time):

#     pr = np.zeros((128, int((end_time - start_time) * 100)))

#     for instrument in pm.PrettyMIDI(midi_file).instruments:
#         for note in instrument.notes:
#             if note.start >= end_time or note.end <= start_time:
#                 continue
#             start = int((note.start - start_time) * 100)
#             end = int((note.end - start_time) * 100)

#             pr[note.pitch, start:end] = 1
    
#     return pr

# start_time, end_time = 0, 30
# beats_seg = beats[np.logical_and(beats >= start_time, beats <= end_time)]
# pr_seg = get_piano_roll(midi_recording, start_time, end_time)

# plt.figure(figsize=(20, 5))
# plt.imshow(pr_seg, aspect='auto', origin='lower', cmap='gray_r')
# for b in beats_seg:
#     plt.axvline((b - start_time) * 100, color='r')
# plt.savefig('test.png')


# =============================================================================\

# from pm2s.features.quantisation import RNNJointQuantisationProcessor

# # Path to the MIDI recording
# midi_recording = '/import/c4dm-05/ll307/datasets/A-MAPS_1.1/MAPS_MUS-alb_esp2_SptkBGAm.mid'

# # Create a beat processor
# processor = RNNJointQuantisationProcessor(None, beat_model_checkpoint='/import/c4dm-05/ll307/workspace/PM2S-draft/mlruns/1/c083be0e407c4ef898896cc8fb23a0f9/checkpoints/epoch=15-val_loss=3.38-val_f1=0.00.ckpt')

# onset_positions, note_values = processor.process(midi_recording)
# print(onset_positions[:100])
# print(note_values[:100])

# =============================================================================

# from pm2s.features.time_signature import RNNTimeSignatureProcessor

# # Path to the MIDI recording
# midi_recording = '/import/c4dm-05/ll307/datasets/A-MAPS_1.1/MAPS_MUS-alb_esp2_SptkBGAm.mid'

# # Create a beat processor
# processor = RNNTimeSignatureProcessor(None, beat_model_checkpoint='/import/c4dm-05/ll307/workspace/PM2S-draft/mlruns/1/c083be0e407c4ef898896cc8fb23a0f9/checkpoints/epoch=15-val_loss=3.38-val_f1=0.00.ckpt')

# ts_changes = processor.process(midi_recording)
# print(ts_changes)

# =============================================================================


# from pm2s.features.key_signature import RNNKeySignatureProcessor

# # Path to the MIDI recording
# midi_recording = '/import/c4dm-05/ll307/datasets/A-MAPS_1.1/MAPS_MUS-alb_esp2_SptkBGAm.mid'

# # Create a beat processor
# processor = RNNKeySignatureProcessor(None, beat_model_checkpoint='/import/c4dm-05/ll307/workspace/PM2S-draft/mlruns/1/c083be0e407c4ef898896cc8fb23a0f9/checkpoints/epoch=15-val_loss=3.38-val_f1=0.00.ckpt')

# ks_changes = processor.process(midi_recording)
# print(ks_changes)


# =============================================================================


from pm2s.features.hand_part import RNNHandPartProcessor

# Path to the MIDI recording
midi_recording = '/import/c4dm-05/ll307/datasets/A-MAPS_1.1/MAPS_MUS-alb_esp2_SptkBGAm.mid'

# Create a beat processor
processor = RNNHandPartProcessor(None, beat_model_checkpoint='/import/c4dm-05/ll307/workspace/PM2S-draft/mlruns/1/c083be0e407c4ef898896cc8fb23a0f9/checkpoints/epoch=15-val_loss=3.38-val_f1=0.00.ckpt')

hand_parts = processor.process(midi_recording)
print(hand_parts)


