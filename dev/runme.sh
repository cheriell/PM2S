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

# Modify the following paths to your own evaluation tool directory
MV2H="/import/c4dm-05/ll307/tools/MV2H/bin"


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
#     --feature 'beat' \
#     # --full_train


# # ========================================================
# # Save model state dict
# # ========================================================
# # Change the model_checkpoint_path to your own trained model checkpoint path, this will save the model to the default path (replacing the pre-trained model state dict)
# python3 save_model.py \
#     --model_checkpoint_path /import/c4dm-05/ll307/workspace/PM2S-draft/mlruns/3/7748838831a24d0fb332b849354851f1/checkpoints/epoch=30-val_loss=3.15-val_f1=0.88.ckpt \
#     --feature 'beat' \
#     # --beat_model_checkpoint ../_model_state_dicts/beat/RNNJointBeatModel_fullTrain.pth

