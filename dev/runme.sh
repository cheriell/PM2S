#!/bin/bash

# ========================================================
# Workspace, dataset, and evaluation tools
# ========================================================
# Modify the following paths to your own workspace
WORKSPACE="/import/c4dm-05/ll307/workspace/PM2S-draft"

# Modify the following paths to your own dataset directory
ASAP="/import/c4dm-05/ll307/datasets/asap-dataset-master"
A_MAPS="/import/c4dm-05/ll307/datasets/A-MAPS_1.1"
CPM="/import/c4dm-05/ll307/datasets/CPM"
ACPAS="/import/c4dm-05/ll307/datasets/ACPAS-dataset"

# # ========================================================
# # Feature preparation 
# # ========================================================
# python3 feature_preparation.py \
#     --dataset_folder $ASAP $A_MAPS $CPM $ACPAS \
#     --feature_folder $WORKSPACE/features \
#     --workers 4 \


# # ========================================================
# # Model training
# # ========================================================
# # feature can be 'beat', 'quantisation', 'time_signature', 'key_signature', 'hand_part'
# python3 train.py \
#     --workspace $WORKSPACE \
#     --ASAP $ASAP \
#     --A_MAPS $A_MAPS \
#     --CPM $CPM \
#     --feature 'time_signature' \
#     # --full_train


# # ========================================================
# # Save model state dict
# # ========================================================
# # Change the model_checkpoint_path to your own trained model checkpoint path, this will save the model to the default path (replacing the pre-trained model state dict)
# python3 save_model.py \
#     --model_checkpoint_path /import/c4dm-05/ll307/workspace/PM2S-draft/mlruns/5/9cd46d4570eb45ca8e72052b479e7ac4/checkpoints/epoch=57-val_loss=0.23-val_f1=0.86.ckpt \
#     --feature 'key_signature' \
#     # --beat_model_checkpoint ../_model_state_dicts/beat/RNNJointBeatModel_fullTrain.pth
