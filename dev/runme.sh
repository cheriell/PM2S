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
#     --transcribed 1 \
#     --transcribed_midi_path $transcribed_midi_path \


# ========================================================
# Model training (features: 'beat' 'quantisation' 'time_signature' 'key_signature' 'hand_part')
# ========================================================
# for feature in 'beat' 'quantisation' 'time_signature' 'key_signature' 'hand_part'
# do
# 	python3 train.py --workspace $WORKSPACE --ASAP $ASAP --A_MAPS $A_MAPS --CPM $CPM --feature $feature --full_train
# done

python3 train.py --workspace $WORKSPACE --ASAP $ASAP --A_MAPS $A_MAPS --CPM $CPM --feature 'beat' --mode 'mixed' --omit_input_feature duration
python3 train.py --workspace $WORKSPACE --ASAP $ASAP --A_MAPS $A_MAPS --CPM $CPM --feature 'beat' --mode 'clean' --omit_input_feature duration
python3 train.py --workspace $WORKSPACE --ASAP $ASAP --A_MAPS $A_MAPS --CPM $CPM --feature 'beat' --mode 'transcribed' --omit_input_feature duration

# ========================================================
# Convert model checkpoint to model state dict
# ========================================================
# Change the model_checkpoint_path to your own trained model checkpoint path. If model_state_dict_path is not specified, this will save the model to the default path (_model_state_dict.pth)
# python3 model_checkpoint_2_state_dict.py \
#     --feature 'beat' \
#     --model_checkpoint_path /import/c4dm-05/ll307/workspace/PM2S-transcribed/mlruns/1/886c4e78262a40a6be39c6d57761b675/checkpoints/epoch=96-val_loss=3.9793-val_f1=0.7616.ckpt \
#     --model_state_dict_path /import/c4dm-05/ll307/repositories/PM2S/pm2s/_model_state_dicts/beat/beat_model_transcribed.pth \

# ========================================================
# Evaluation
# ========================================================
# This will evaluate on both clean performance MIDI and transcribed MIDI (from audio recordings) together.
# python3 evaluate.py \
#     --feature beat \
#     --workspace $WORKSPACE \
#     --model_state_dict_path /import/c4dm-05/ll307/repositories/PM2S/pm2s/_model_state_dicts/beat/beat_model_temp.pth \
#     --device cuda:0 \

# python3 evaluate.py \
#     --feature time_signature \
#     --workspace $WORKSPACE \
#     --model_state_dict_path /import/c4dm-05/ll307/repositories/PM2S/_model_state_dicts/time_signature/CNNTimeSignatureModel.pth \
#     --device cuda:0 \

# python3 evaluate.py \
#     --feature key_signature \
#     --workspace $WORKSPACE \
#     --model_state_dict_path /import/c4dm-05/ll307/repositories/PM2S/_model_state_dicts/key_signature/RNNKeySignatureModel.pth \
#     --device cuda:0 \

# python3 evaluate.py \
#     --feature hand_part \
#     --workspace $WORKSPACE \
#     --model_state_dict_path /import/c4dm-05/ll307/repositories/PM2S/_model_state_dicts/hand_part/RNNHandPartModel.pth \
#     --device cuda:0 \

# python3 evaluate.py \
#     --feature quantisation \
#     --workspace $WORKSPACE \
#     --model_state_dict_path /import/c4dm-05/ll307/repositories/PM2S/_model_state_dicts/quantisation/RNNJointQuantisationModel.pth \
#     --device cuda:0 \

# For MV2H evaluation, we use:
# - `crnn_joint_pm2s()` to generate the score MIDI file, using the trained model. We didn't include time signature in the score MIDI file for Mv2H evaluation.
# - `evaluate_midi_mv2h.sh` to evaluate the performance of the generated score MIDI file, using the mv2h evaluation tool.
