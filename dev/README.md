# Model training instructions

Please install the requirements before training the model by running:

    cd dev
    pip install requirements_train.txt

We use the [A-MAPS](http://c4dm.eecs.qmul.ac.uk/ycart/a-maps.html), [Classical Piano Midi](https://github.com/cheriell/ClassicalPianoMIDI-dataset), and [ASAP](https://github.com/fosfrancesco/asap-dataset) datasets for developing the model. You can download them from the links.

Change the paths for workspace and datasets in `runme.sh` and you can train the model using the shell script.

