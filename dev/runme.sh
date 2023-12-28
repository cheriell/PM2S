#!/bin/bash

# ========================================================
# Workspace, dataset, and evaluation tools
# ========================================================
# Modify the following paths to your own workspace
WORKSPACE="/import/c4dm-05/ll307/workspace/PM2S-timesigcorrection"

# Modify the following paths to your own dataset directory
ASAP="/import/c4dm-05/ll307/datasets/asap-dataset-note_alignments"
A_MAPS="/import/c4dm-05/ll307/datasets/A-MAPS_1.1"
CPM="/import/c4dm-05/ll307/datasets/ClassicalPianoMIDI-dataset"
ACPAS="/import/c4dm-05/ll307/datasets/ACPAS-dataset"

# This is the path to the transcribed midi files, from the ACPAS audio recordings, 
# using high-resolution piano transcription model.
transcribed_midi_path="/import/c4dm-05/ll307/repositories/pipeline-A2S/transcribed_midi"

# ========================================================
# Feature preparation 
# ========================================================
# python3 feature_preparation.py \
#     --dataset_folder $ASAP $A_MAPS $CPM $ACPAS \
#     --feature_folder $WORKSPACE/features \
#     --workers 4 \
#     --transcribed False \
#     --transcribed_midi_path $transcribed_midi_path \


# ========================================================
# Model training
# ========================================================
# feature can be 'beat', 'quantisation', 'time_signature', 'key_signature', 'hand_part'
# python3 train.py \
#     --workspace $WORKSPACE \
#     --ASAP $ASAP \
#     --A_MAPS $A_MAPS \
#     --CPM $CPM \
#     --feature 'time_signature' \
#     --full_train


# ========================================================
# Save model state dict
# ========================================================
# Change the model_checkpoint_path to your own trained model checkpoint path, this will save the model to the default path (replacing the pre-trained model state dict)
python3 save_model.py \
    --model_checkpoint_path /import/c4dm-05/ll307/workspace/PM2S-timesigcorrection/mlruns/1/7e3324f3fa4b466aac078fedab72e467/checkpoints/epoch=32-val_loss=0.58-val_f1=0.27.ckpt \
    --feature 'time_signature' \
    --save_to_path nohup.out.timesig_model.pth
    # --beat_model_checkpoint ../_model_state_dicts/beat/RNNJointBeatModel_fullTrain.pth
