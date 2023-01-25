# Performance MIDI-to-Score Conversion

This is the official implementation for our ISMIR 2022 paper [Performance MIDI-to-Score Conversion by Neural Beat Tracking](https://www.turing.ac.uk/research/publications/performance-midi-score-conversion-neural-beat-tracking).

(Pre-trained model & training script to be updated by 5th February)

## Environments

The implementation is based on Python 3.8 and PyTorch 1.12.0. Other versions should work, but they are not fully tested. A list of python dependencies is available in `requirements.txt`, run:

    pip install -r requirements.txt

## Running instructions

To use the pre-trained model for prediction, please follow the instructions in `demo.ipynb`. To train a model from scratch, please refer to the instructions in `dev/README.md`.
