import torch
import torch.nn.functional as F

from pm2s.constants import resolution

def get_in_features(omit_feature=None):
    """
    Return the number of input features for the proposed model
    
    Parameters
    ----------
    omit_feature: (str)
        Which feature to omit. Default is None.
        Can be 'pitch', 'onset', 'duration', 'velocity'.
    
    Returns
    -------
    in_features: (int)
        Dimension of input features.
    """
    
    features_pitch = 128
    features_onset = int(4.0 / resolution) + 1  # maximum 4s for onset shift
    features_duration = 1   # 1 for duration in raw value
    features_velocity = 1   # 1 for velocity in raw value and normalised to 0-1
    
    if omit_feature is None:
        in_features = features_pitch + features_onset + features_duration + features_velocity
    elif omit_feature == 'pitch':
        in_features = features_onset + features_duration + features_velocity
    elif omit_feature == 'onset':
        in_features = features_pitch + features_duration + features_velocity
    elif omit_feature == 'duration':
        in_features = features_pitch + features_onset + features_velocity
    elif omit_feature == 'velocity':
        in_features = features_pitch + features_onset + features_duration
    else:
        raise ValueError('Invalid omit_feature')
    
    return in_features

def encode_note_sequence(note_seq, omit_feature=None):
    """
    Encode note sequence.

    Parameters
    ----------
    note_seq: (torch.Tensor)
        Note sequence tensor.
    omit_feature: (str)
        Which feature to omit. Default is None.
        Can be 'pitch', 'onset', 'duration', 'velocity'.

    Returns
    -------
    feature: (torch.Tensor)
        Encoded note sequence tensor.
    """
    x_pitch = note_seq[:, :, 0]
    x_onset = note_seq[:, :, 1]
    x_duration = note_seq[:, :, 2]
    x_velocity = note_seq[:, :, 3]

    # pitch
    pitch_onehot = F.one_hot(x_pitch.long(), 128).float()

    # onset
    onset_shift = x_onset[:,1:] - x_onset[:,:-1]
    onset_shift = torch.cat([torch.zeros_like(onset_shift[:,0:1]), onset_shift], dim=1)
    # max onset shift to 4s
    onset_shift = torch.clamp(onset_shift, 0, 4)
    # to one hot encoding
    onset_shift_onehot = F.one_hot(torch.round((onset_shift/resolution)).long(), int(4/resolution+1)).float()

    # duration
    duration_raw = x_duration.float().unsqueeze(2)

    # velocity
    velocity_norm = x_velocity.float().unsqueeze(2) / 127.0

    if omit_feature is None:
        feature = torch.cat([pitch_onehot, onset_shift_onehot, duration_raw, velocity_norm], dim=2)
    elif omit_feature == 'pitch':
        feature = torch.cat([onset_shift_onehot, duration_raw, velocity_norm], dim=2)
    elif omit_feature == 'onset':
        feature = torch.cat([pitch_onehot, duration_raw, velocity_norm], dim=2)
    elif omit_feature == 'duration':
        feature = torch.cat([pitch_onehot, onset_shift_onehot, velocity_norm], dim=2)
    elif omit_feature == 'velocity':
        feature = torch.cat([pitch_onehot, onset_shift_onehot, duration_raw], dim=2)
    else:
        raise ValueError('Invalid omit_feature')

    return feature
    