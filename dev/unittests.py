# Unit test for the dev module

import unittest
import os
import sys
import shutil
import tempfile
import subprocess
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.append('../')


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


class TestDev(unittest.TestCase):

    def test_time_signature_distribution(self):
        from data.dataset_time_signature import TimeSignatureDataset

        split = 'test'

        time_signature_dataset = TimeSignatureDataset(
            workspace=WORKSPACE, 
            split=split,
        )
        print('dataset size:', len(time_signature_dataset))

        timeSigs_framewise_all = []
        timeSigs_itemwise_all = []
        for i in range(len(time_signature_dataset)):
            _, timeSigs, length = time_signature_dataset[i]

            timeSigs_framewise_all.append(timeSigs[:length])
            timeSigs_itemwise_all.append(np.mean(timeSigs[:length]))

        timeSigs_framewise_all = np.concatenate(timeSigs_framewise_all)
        timeSigs_itemwise_all = np.array(timeSigs_itemwise_all)

        ratio = timeSigs_framewise_all.sum() / len(timeSigs_framewise_all)
        print('3/4 ratio:', ratio)
        
        import matplotlib.pyplot as plt
        plt.hist(timeSigs_itemwise_all, bins=20)
        plt.savefig('nohup.out.time_signature_distribution.png')


if __name__ == '__main__':
    unittest.main()