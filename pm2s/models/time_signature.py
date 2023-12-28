import torch
import torch.nn as nn

from pm2s.models.utils import get_in_features, encode_note_sequence
from pm2s.models.blocks import ConvBlock, GRUBlock, LinearOutput
from pm2s.models.beat import RNNJointBeatModel
# from pm2s.constants import tsNumeVocabSize, tsDenoVocabSize, timeSigVocabSize


class CNNTimeSignatureModel(nn.Module):

    def __init__(self, hidden_size=256):
        super().__init__()

        in_features = get_in_features()

        self.convs = ConvBlock(in_features=in_features, hidden_size=hidden_size)

        self.out_ts = LinearOutput(in_features=hidden_size, out_features=1, activation_type='sigmoid')
        
    def forward(self, x):
        # x: (batch_size, seq_len, len(features)==4)
        x = encode_note_sequence(x)

        x = self.convs(x)  # (batch_size, seq_len, hidden_size)

        y_ts = self.out_ts(x)  # (batch_size, seq_len, 1)
        y_ts = y_ts.squeeze(2)  # (batch_size, seq_len)

        return y_ts

