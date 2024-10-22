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
--------

## to-do's
- [X] Data Download & Preparing
  - [X] AudioCaps, Jinju Kim
  - [X] Clotho, Jinju Kim
- [ ] Baseline Model Reconstruction
  - [ ] MS CLAP '23
  - [ ] Pengi-Enc
  - [ ] LAION CLAP
  - [ ] MS CLAP '22

