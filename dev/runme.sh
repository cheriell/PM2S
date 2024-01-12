#!/bin/bash

# ========================================================
# Workspace, dataset, and evaluation tools
# ========================================================
# Modify the following paths to your own workspace
WORKSPACE="/import/c4dm-05/ll307/workspace/PM2S-transcribed"

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
#     --transcribed 0 \
#     # --transcribed_midi_path $transcribed_midi_path \


# ========================================================
# Model training (features: 'beat' 'quantisation' 'time_signature' 'key_signature' 'hand_part')
# ========================================================
# for feature in 'beat' 'quantisation' 'time_signature' 'key_signature' 'hand_part'
# do
# 	python3 train.py --workspace $WORKSPACE --ASAP $ASAP --A_MAPS $A_MAPS --CPM $CPM --feature $feature --full_train
# done
# python3 train.py --workspace $WORKSPACE --ASAP $ASAP --A_MAPS $A_MAPS --CPM $CPM --feature time_signature --full_train

# ========================================================
# Convert model checkpoint to model state dict
# ========================================================
# Change the model_checkpoint_path to your own trained model checkpoint path. If model_state_dict_path is not specified, this will save the model to the default path (_model_state_dict.pth)
# python3 model_checkpoint_2_state_dict.py \
#     --feature 'time_signature' \
#     --model_checkpoint_path /import/c4dm-05/ll307/workspace/PM2S-transcribed/mlruns/1/8eb6656db9a94a1d998b1c61a72545f9/checkpoints/epoch=84-val_loss=0.5156-val_f1=0.2704.ckpt \
#     --model_state_dict_path ../_model_state_dicts_transcribed/time_signature/CNNTimeSignatureModel_1.pth


# ========================================================
# Evaluation
# ========================================================
# This will evaluate on both clean performance MIDI and transcribed MIDI (from audio recordings) together.
python3 evaluate.py \
    --feature beat \
    --workspace $WORKSPACE \
    --model_state_dict_path /import/c4dm-05/ll307/repositories/PM2S/_model_state_dicts/beat/RNNJointBeatModel.pth \
    --device cuda:0 \

python3 evaluate.py \
    --feature time_signature \
    --workspace $WORKSPACE \
    --model_state_dict_path /import/c4dm-05/ll307/repositories/PM2S/_model_state_dicts/time_signature/CNNTimeSignatureModel.pth \
    --device cuda:0 \

python3 evaluate.py \
    --feature key_signature \
    --workspace $WORKSPACE \
    --model_state_dict_path /import/c4dm-05/ll307/repositories/PM2S/_model_state_dicts/key_signature/RNNKeySignatureModel.pth \
    --device cuda:0 \

python3 evaluate.py \
    --feature hand_part \
    --workspace $WORKSPACE \
    --model_state_dict_path /import/c4dm-05/ll307/repositories/PM2S/_model_state_dicts/hand_part/RNNHandPartModel.pth \
    --device cuda:0 \

