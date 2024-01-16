# Performance MIDI-to-Score Conversion

This is the official implementation for our ISMIR 2022 paper [Performance MIDI-to-Score Conversion by Neural Beat Tracking](https://www.turing.ac.uk/research/publications/performance-midi-score-conversion-neural-beat-tracking).

## Environments

The implementation is based on Python 3.8 and PyTorch 1.12.0. Other versions can work, but they are not tested. A list of python dependencies is available in `requirements.txt`, run:

    pip install -r requirements.txt

## Running instructions

To use the pre-trained model for prediction, please follow the instructions in `demo.ipynb`. 

To train a model from scratch, please refer to the instructions in `dev/README.md`.

## Cite

> L. Liu, Q. Kong, V. Morfi and E. Benetos, "Performance MIDI-to-Score Conversion by Neural Beat Tracking" in Proceedings of the 23rd International Society for Music Information Retrieval Conference, Bengaluru, India, Dec 2022.
