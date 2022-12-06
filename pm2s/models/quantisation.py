import torch.nn as nn
import torch

from pm2s.constants import omVocab, nvVocab
from pm2s.models.utils import get_in_features, encode_note_sequence
from pm2s.models.blocks import ConvBlock, GRUBlock, LinearOutput
from pm2s.models.beat import RNNJointBeatModel

class RNNJointQuantisationModel(nn.Module):

    def __init__(self, beat_model_checkpoint, hidden_size=512):

        super().__init__()

        in_features = get_in_features()

        self.conv_onset = ConvBlock(in_features=in_features)
        self.conv_value = ConvBlock(in_features=in_features)

        self.gru_onset = GRUBlock(in_features=hidden_size + 1 + nvVocab) # +1 for beat
        self.gru_value = GRUBlock(in_features=hidden_size)

        self.out_onset = LinearOutput(in_features=hidden_size, out_features=omVocab, activation_type='softmax')
        self.out_value = LinearOutput(in_features=hidden_size, out_features=nvVocab, activation_type='softmax')

        # load beat model and freeze
        # debugging
        self.beat_model = RNNJointBeatModel()
        # self.beat_model = RNNJointBeatModel.load_state_dict(beat_model_checkpoint)
        for param in self.beat_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        # x: (batch_size, seq_len, len(features)==4)

        x = encode_note_sequence(x)
        
        x_conv_onset = self.conv_onset(x)  # (batch_size, seq_len, hidden_size)
        x_conv_value = self.conv_value(x)  # (batch_size, seq_len, hidden_size)

        x_gru_value = self.gru_value(x_conv_value) # (batch_size, seq_len, hidden_size)
        y_value = self.out_value(x_gru_value) # (batch_size, seq_len, noteValueVocab)

        y_beat, y_downbeat, _ = self.beat_model(x)  # (batch_size, seq_len)
        x_concat_onset = torch.cat((x_conv_onset, y_beat.unsqueeze(2), y_value), dim=2) # (batch_size, seq_len, hidden_size + 1 + noteValueVocab)
        x_gru_onset = self.gru_onset(x_concat_onset) # (batch_size, seq_len, hidden_size)
        y_onset = self.out_onset(x_gru_onset) # (batch_size, seq_len, onsetVocab)

        return y_beat, y_downbeat, y_onset, y_value





