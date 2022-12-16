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

# ========================================================
# Feature preparation 
# ========================================================
python3 feature_preparation.py \
    --dataset_folder $ASAP $A_MAPS $CPM $ACPAS \
    --feature_folder $WORKSPACE/features \
    --workers 4 \

# ========================================================
# Model training
# ========================================================
# feature can be 'beat', 'quant', 'time_sig', 'key_sig', 'hand'
python3 train.py \
    --workspace $WORKSPACE \
    --ASAP $ASAP \
    --A_MAPS $A_MAPS \
    --CPM $CPM \
    --feature 'beat' \
    --full_train False

# ========================================================
# Save model state dict
# ========================================================
# Change the model_checkpoint_path to your own trained model checkpoint path, and model_save_path to where you want to save the model state dict
python3 save_model.py \
    --model_checkpoint_path /import/c4dm-05/ll307/workspace/PM2S-draft/mlruns/1/4e6c6172da074a3ca97fbb569737cc03/checkpoints/epoch=47-val_loss=2.13-val_f1=0.92.ckpt \
    --model_save_path ../_model_state_dicts/beat/RNNJointBeatModel.pth \
    --feature 'beat'

# ========================================================
# Model evaluation
# ========================================================
# By default, the evaluation script will use the pre-trianed model state dict.
# If you want to evalute your own trained model, change the model state dict paths to the path of your own trained model state dict.
python3 evaluation.py \
    --ACPAS $ACPAS \
    --feature 'beat' \
    # --model_state_dict_path_beat path/to/beat/model.pth
