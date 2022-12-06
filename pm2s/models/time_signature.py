import torch.nn as nn

from pm2s.models.utils import get_in_features, encode_note_sequence
from pm2s.models.blocks import ConvBlock, GRUBlock, LinearOutput
from pm2s.constants import tsNumeVocabSize, tsDenoVocabSize


class RNNTimeSignatureModel(nn.Module):

    def __init__(self, hidden_size=512):
        super().__init__()

        in_features = get_in_features()

        self.convs = ConvBlock(in_features=in_features)

        self.gru = GRUBlock(in_features=hidden_size)

        self.out_td = LinearOutput(in_features=hidden_size, out_features=tsDenoVocabSize, activation_type='softmax')
        self.out_tn = LinearOutput(in_features=hidden_size, out_features=tsNumeVocabSize, activation_type='softmax')

    def forward(self, x):
        # x: (batch_size, seq_len, len(features)==4)
        x = encode_note_sequence(x)

        x = self.convs(x)  # (batch_size, seq_len, hidden_size)

        x_gru = self.gru(x)  # (batch_size, seq_len, hidden_size)

        y_td = self.out_td(x_gru)  # (batch_size, seq_len, 1)
        y_tn = self.out_tn(x_gru)  # (batch_size, seq_len, 1)

        return y_td, y_tn

