import torch.nn as nn

from pm2s.models.blocks import ConvBlock, GRUBlock, LinearOutput
from pm2s.constants import keyVocabSize
from pm2s.models.utils import get_in_features, encode_note_sequence

class RNNKeySignatureModel(nn.Module):

    def __init__(self, hidden_size=512):
        super().__init__()

        in_features = get_in_features()

        self.convs = ConvBlock(in_features=in_features)

        self.gru = GRUBlock(in_features=hidden_size)

        self.out = LinearOutput(in_features=hidden_size, out_features=keyVocabSize, activation_type='softmax')

    def forward(self, x):
        # x: (batch_size, seq_len, len(features)==4)
        x = encode_note_sequence(x)

        x = self.convs(x) # (batch_size, seq_len, hidden_size)
        x = self.gru(x) # (batch_size, seq_len, hidden_size)
        y = self.out(x) # (batch_size, seq_len, keyVocabSize)

        return y