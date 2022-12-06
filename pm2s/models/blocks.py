import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_features, hidden_size=512, kernel_size=9, dropout=0.15):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=hidden_size // 4,
                kernel_size=(kernel_size, in_features),
                padding=(kernel_size // 2, 0),
            ),
            nn.BatchNorm2d(hidden_size // 4),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(
                in_channels=hidden_size // 4,
                out_channels=hidden_size // 2,
                kernel_size=(kernel_size, 1),
                padding=(kernel_size // 2, 0),
            ),
            nn.BatchNorm2d(hidden_size // 2),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(
                in_channels=hidden_size // 2,
                out_channels=hidden_size,
                kernel_size=(kernel_size, 1),
                padding=(kernel_size // 2, 0),
            ),
            nn.BatchNorm2d(hidden_size),
            nn.ELU(),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        # x: (batch_size, sequence_length, in_features)
        x = x.unsqueeze(1)   # (batch_size, 1, sequence_length, in_features)
        x = self.conv(x)   # (batch_size, hidden_size, sequence_length, 1)
        x = x.squeeze(3).transpose(1, 2)  # (batch_size, sequence_length, hidden_size)
        return x


class GRUBlock(nn.Module):
    def __init__(self, in_features, hidden_size=512, gru_layers=2, dropout=0.15):
        super().__init__()

        self.grus_beat = nn.GRU(
            input_size=in_features,
            hidden_size=hidden_size,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.linear = nn.Linear(hidden_size*2, hidden_size)

    def forward(self, x):
        # x: (batch_size, sequence_length, hidden_size)

        x, _ = self.grus_beat(x)  # (batch_size, sequence_length, hidden_size*2)
        x = self.linear(x)  # (batch_size, sequence_length, hidden_size)

        return x


class LinearOutput(nn.Module):
    def __init__(self, in_features, out_features, activation_type='sigmoid', dropout=0.15):
        super().__init__()

        self.activation_type = activation_type

        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(in_features, out_features)

        if activation_type == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation_type == 'softmax':
            self.activation = nn.LogSoftmax(dim=2)
        elif activation_type == 'softplus':
            self.activation = nn.Softplus()

    def forward(self, x):
        # x: (batch_size, sequence_length, in_features)

        x = self.dropout(x)  # (batch_size, sequence_length, in_features)
        x = self.linear(x)  # (batch_size, sequence_length, out_features)
        x = self.activation(x)  # (batch_size, sequence_length, out_features)

        return x