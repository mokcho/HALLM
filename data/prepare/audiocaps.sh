#!/bin/bash

mkdir -p ./data/AudioCaps/annotation
mkdir -p ./data/AudioCaps/audio

ANNOT_PATH="./data/AudioCaps/annotation"

TRAIN_URL="https://raw.githubusercontent.com/cdjkim/audiocaps/refs/heads/master/dataset/train.csv"
TEST_URL="https://raw.githubusercontent.com/cdjkim/audiocaps/refs/heads/master/dataset/test.csv"
VAL_URL="https://raw.githubusercontent.com/cdjkim/audiocaps/refs/heads/master/dataset/val.csv"

mkdir -p "$ANNOT_PATH"

wget -v $TRAIN_URL -O "$ANNOT_PATH/train.csv"
wget -v $TEST_URL -O "$ANNOT_PATH/test.csv"
wget -v $VAL_URL -O "$ANNOT_PATH/val.csv"

echo "Download complete. Files are saved in $ANNOT_PATH"

echo "Downloading audio files from Youtube"
python audiocaps.py