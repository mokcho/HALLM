#!/bin/bash

mkdir -p ./data/Clotho/caption
mkdir -p ./data/Clotho/audio

TARGET_DIR="./data/Clotho"
CAPTION_DIR="./data/Clotho/caption"

DEV_URL="https://zenodo.org/records/3490684/files/clotho_audio_development.7z?download=1"
EVAL_URL="https://zenodo.org/records/3490684/files/clotho_audio_evaluation.7z?download=1"
VAL_URL = "https://zenodo.org/records/3490684/files/clotho_audio_validation.7z?download=1"

DEV_CAP_URL="https://zenodo.org/records/3490684/files/clotho_captions_development.csv?download=1"
EVAL_CAP_URL="https://zenodo.org/records/3490684/files/clotho_captions_evaluation.csv?download=1"
VAl_CAP_URL = "https://zenodo.org/records/3490684/files/clotho_captions_validation.csv?download=1"

DEV_META_URL="https://zenodo.org/records/3490684/files/clotho_metadata_development.csv?download=1"
EVAL_META_URL="https://zenodo.org/records/3490684/files/clotho_metadata_evaluation.csv?download=1"
VAL_META_URL="https://zenodo.org/records/3490684/files/clotho_metadata_validation.csv?download=1"


# Download the file
echo "Downloading Clotho dataset..."
wget -O "$TARGET_DIR/dev_metadata.csv" "$EVAL_META_URL"
wget -O "$TARGET_DIR/eval_metadata.csv" "$DEV_META_URL"
wget -O "$TARGET_DIR/val_metadata.csv" "$VAL_META_URL"

wget -O "$CAPTION_DIR/eval.csv" "$EVAL_CAP_URL"
wget -O "$CAPTION_DIR/dev.csv" "$DEV_CAP_URL"
wget -O "$CAPTION_DIR/val.csv" "$VAL_CAP_URL"

wget -O "$TARGET_DIR/clotho_audio_development.7z" "$DEV_URL"
wget -O "$TARGET_DIR/clotho_audio_evaluation.7z" "$EVAL_URL"
wget -O "$TARGET_DIR/clotho_audio_validation.7z" "$VAL_URL"


# Inform the user
if [ $? -eq 0 ]; then
    echo "Download completed successfully."
    echo "File saved to: $OUTPUT_FILE"
else
    echo "Download failed."
fi

# Install p7zip depending on OS
brew install p7zip #macOS using homebrew
# sudo apt-get install p7zip-full #linux/ubuntu

7z x "$TARGET_DIR/clotho_audio_development.7z" -o./data/Clotho/audio/
7z x "$TARGET_DIR/clotho_audio_evaluation.7z" -o./data/Clotho/audio/
7z x "$TARGET_DIR/clotho_audio_validation.7z" -o./data/Clotho/audio/


