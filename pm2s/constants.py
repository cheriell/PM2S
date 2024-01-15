import os

# ========== data representation related constants ==========
## quantisation resolution
resolution = 0.01  # quantization resolution: 0.01s = 10ms
tolerance = 0.05  # tolerance for beat alignment: 0.05s = 50ms
ibiVocab = int(4 / resolution) + 1  # vocabulary size for inter-beat-interval: 4s = 4/0.01s + 1, index 0 is ignored during training

# ========== post-processing constants ==========
min_bpm = 40
max_bpm = 240
ticks_per_beat = 240

# =========== time signature definitions ===========
# Update Dec 2023: change to use binary classifications for time signature prediction
# 0: 4-based time signature (such as 4/4, 2/4), 1: 3-based time signature (such as 3/4, 6/8)
# we can then infer the numerator and denominator from the beats and downbeats predictions.

# =========== key signature definitions ==========
# key in sharps in mido
keySharps2Name = {0: 'C', 1: 'G', 2: 'D', 3: 'A', 4: 'E', 5: 'B', 6: 'F#',
                  7: 'C#m', 8: 'G#m', 9: 'D#m', 10: 'Bbm', 11: 'Fm', 12: 'Cm',
                  -11: 'Gm', -10: 'Dm', -9: 'Am', -8: 'Em', -7: 'Bm', -6: 'F#m',
                  -5: 'Db', -4: 'Ab', -3: 'Eb', -2: 'Bb', -1: 'F'}
keyName2Sharps = dict([(name, sharp) for sharp, name in keySharps2Name.items()])
# key in numbers in pretty_midi
keyNumber2Name = [
    'C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B',
    'Cm', 'C#m', 'Dm', 'D#m', 'Em', 'Fm', 'F#m', 'Gm', 'G#m', 'Am', 'Bbm', 'Bm',
]
keyName2Number = dict([(name, number) for number, name in enumerate(keyNumber2Name)])
keySharps2Number = dict([(sharp, keyName2Number[keySharps2Name[sharp]]) for sharp in keySharps2Name.keys()])
keyNumber2Sharps = dict([(number, keyName2Sharps[keyNumber2Name[number]]) for number in range(len(keyNumber2Name))])
keyVocabSize = len(keySharps2Name) // 2  # ignore minor keys in key signature prediction!

# =========== onset musical & note value definitions ===========
# proposed model
N_per_beat = 24  # 24 resolution per beat
max_note_value = 4 * N_per_beat  # 4 beats
omVocab = N_per_beat  # onset musical vocabulary size
nvVocab = max_note_value + 1  # note value vocabulary size, 0 is ignored during training

# =========== model state dicts ===========
model_state_dict_paths = {
    'beat': {
        'state_dict_path': os.path.join(os.path.dirname(__file__), '../_model_state_dicts/beat/RNNJointBeatModel.pth'),
        'zenodo_path': '',
    },
    'quantisation': {
        'state_dict_path': '_model_state_dicts/quantisation/RNNJointQuantisationModel.pth',
        'zenodo_path': '',
    },
    'hand_part': {
        'state_dict_path': '_model_state_dicts/hand_part/RNNHandPartModel.pth',
        'zenodo_path': '',
    },
    'key_signature': {
        'state_dict_path': '_model_state_dicts/key_signature/RNNKeySignatureModel.pth',
        'zenodo_path': '',
    },
    'time_signature': {
        'state_dict_path': '_model_state_dicts/time_signature/CNNTimeSignatureModel.pth',
        'zenodo_path': '',
    },
}