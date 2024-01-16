#!/usr/bin/env bash

if [ "$#" -ne 3 ]; then
    echo "USAGE: evaluate_midi.sh ref.mid transcription.mid MV2H_path"
    exit 1
fi

java -cp $3 mv2h.tools.Converter -i $1 >$1.conv.txt
java -cp $3 mv2h.tools.Converter -i $2 >$2.conv.txt
java -cp $3 mv2h.Main -g $1.conv.txt -t $2.conv.txt
rm $1.conv.txt $2.conv.txt