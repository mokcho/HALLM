# HALLM
Detecting Hallucination in ALMs (Name is temporary)

Group #34 Project for 11-785 : Intro to Deep Learning at CMU

--------

# Data Download & Process

## AudioCaps

This downloads annotations of [AudioCaps](https://github.com/cdjkim/audiocaps) and corresponding audio files fron Youtube, trims and saves in 44.1khz wav file to ./data/AudioCaps

```
./data/prepare/AudioCaps.sh
```

## Clotho

This downloads and unzips audio files, caption files, metadata of [Clotho](https://github.com/audio-captioning/clotho-dataset) to ./data/Clotho. Audio files of Clotho are already equally sampled at 44.1khz.

```
./data/prepare/Clotho.sh
```
## AudioCaps Entailment and Clotho Entailment

download and move gpt4 generated entailment .csv under data/AudioCaps/entailment and data/Clotho/entailment.
```
python ./data/prepare/process.py --data clotho --data_dir ./data
```

# Classifier Training

## Pengi-enc

1. Clone pengi from [microsoft/Pengi](https://github.com/microsoft/Pengi)
```
cd ./models
git clone https://github.com/microsoft/Pengi.git
```
follow Pengi's readme to obtain checkpoint of model

2. Train with main.py
```
# set your own model_cfg in ./configs for trainings
python main.py --model pengi --model_cfg ./configs/pengi_linear_classifier.yaml --classifier linear --data_dir ./data
```



--------

## to-do's
- [X] Data Download & Preparing
  - [X] AudioCaps, Jinju Kim
  - [X] Clotho, Jinju Kim
- [ ] Baseline Model Reconstruction
  - [ ] MS CLAP '23
  - [X] Pengi-Enc, Jinju Kim
  - [ ] LAION CLAP
  - [ ] MS CLAP '22

