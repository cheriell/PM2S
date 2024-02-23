import torch.nn as nn
import torch

from pm2s.constants import omVocab, nvVocab, model_state_dict_paths
from pm2s.models.utils import get_in_features, encode_note_sequence
from pm2s.models.blocks import ConvBlock, GRUBlock, LinearOutput
from pm2s.models.beat import RNNJointBeatModel

class RNNJointQuantisationModel(nn.Module):

    def __init__(self, hidden_size=512, beat_model_state_dict=model_state_dict_paths['beat']['state_dict_path'], beat_type='estimated'):

        super().__init__()

        in_features = get_in_features()

        self.conv_onset = ConvBlock(in_features=in_features)
        self.conv_value = ConvBlock(in_features=in_features)

        self.gru_onset = GRUBlock(in_features=hidden_size + 1 + nvVocab) # +1 for beat
        self.gru_value = GRUBlock(in_features=hidden_size)

        self.out_onset = LinearOutput(in_features=hidden_size, out_features=omVocab, activation_type='softmax')
        self.out_value = LinearOutput(in_features=hidden_size, out_features=nvVocab, activation_type='softmax')

        # load beat model and freeze its parameters
        self.beat_model = RNNJointBeatModel()
        self.beat_model.load_state_dict(torch.load(beat_model_state_dict))
        for name, param in self.beat_model.named_parameters():
            param.requires_grad = False

        self.beat_type = beat_type

    def forward(self, x, x_beat=None):
        # Get x_beat
        if x_beat is not None:
            if self.beat_type == 'estimated':
                x_beat, _, _ = self.beat_model(x)  # (batch_size, seq_len)
            elif self.beat_type == 'ground_truth':
                pass
            elif self.beat_type == 'mixed':
                if torch.rand(1) < 0.5:
                    x_beat, _, _ = self.beat_model(x)
            else:
                raise ValueError('Invalid beat type.')
        else:
            x_beat, _, _ = self.beat_model(x)

        # x: (batch_size, seq_len, len(features)==4)
        x = encode_note_sequence(x)
        
        x_conv_onset = self.conv_onset(x)  # (batch_size, seq_len, hidden_size)
        x_conv_value = self.conv_value(x)  # (batch_size, seq_len, hidden_size)

        x_gru_value = self.gru_value(x_conv_value) # (batch_size, seq_len, hidden_size)
        y_value = self.out_value(x_gru_value) # (batch_size, seq_len, noteValueVocab)
        
        x_concat_onset = torch.cat((x_conv_onset, x_beat.unsqueeze(2), y_value), dim=2) # (batch_size, seq_len, hidden_size + 1 + noteValueVocab)
        x_gru_onset = self.gru_onset(x_concat_onset) # (batch_size, seq_len, hidden_size)
        y_onset = self.out_onset(x_gru_onset) # (batch_size, seq_len, onsetVocab)

        y_onset = y_onset.transpose(1, 2) # (batch_size, onsetVocab, seq_len)
        y_value = y_value.transpose(1, 2) # (batch_size, noteValueVocab, seq_len)

        # return y_beat, y_downbeat, y_onset, y_value
        return y_onset, y_value





