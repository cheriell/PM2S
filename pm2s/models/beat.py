
import torch.nn as nn

from pm2s.constants import ibiVocab
from pm2s.models.utils import get_in_features, encode_note_sequence
from pm2s.models.blocks import ConvBlock, GRUBlock, LinearOutput

class RNNJointBeatModel(nn.Module):

    def __init__(self, hidden_size=512):
        super().__init__()

        in_features = get_in_features()

        self.convs = ConvBlock(in_features=in_features)

        self.gru_beat = GRUBlock(in_features=hidden_size)
        self.gru_downbeat = GRUBlock(in_features=hidden_size)
        self.gru_tempo = GRUBlock(in_features=hidden_size)

        self.out_beat = LinearOutput(in_features=hidden_size, out_features=1, activation_type='sigmoid')
        self.out_downbeat = LinearOutput(in_features=hidden_size, out_features=1, activation_type='sigmoid')
        self.out_tempo = LinearOutput(in_features=hidden_size, out_features=ibiVocab, activation_type='softmax')

    def forward(self, x):
        # x: (batch_size, seq_len, len(features)==4)
        x = encode_note_sequence(x)

        x = self.convs(x)  # (batch_size, seq_len, hidden_size)

        x_gru_beat = self.gru_beat(x)  # (batch_size, seq_len, hidden_size)
        x_gru_downbeat = self.gru_downbeat(x_gru_beat)  # (batch_size, seq_len, hidden_size)
        x_gru_tempo = self.gru_tempo(x_gru_downbeat)  # (batch_size, seq_len, hidden_size)

        y_beat = self.out_beat(x_gru_beat)  # (batch_size, seq_len, 1)
        y_downbeat = self.out_downbeat(x_gru_downbeat)  # (batch_size, seq_len, 1)
        y_tempo = self.out_tempo(x_gru_tempo)  # (batch_size, seq_len, ibiVocab)

        # squeeze and transpose
        y_beat = y_beat.squeeze(2)  # (batch_size, seq_len)
        y_downbeat = y_downbeat.squeeze(2)  # (batch_size, seq_len)
        y_tempo = y_tempo.transpose(1, 2)  # (batch_size, ibiVocab, seq_len)

        return y_beat, y_downbeat, y_tempo


