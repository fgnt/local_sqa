# Localizable Speech Quality Assessment

## About
Conventional speech quality assessment (SQA) models predict a single score for the entire audio clip.
However, in many applications, it might be desirable to have quality estimates of finer temporal resolution.
This repository implements a framework for training SQA models that predict frame-level quality scores.
Building upon SSL-MOS [1], the idea is to add a consistency constraint that brings the encoder outputs of audio segments *within* and *detached* from the context close to each other in the embedding space.

[1] Erica Cooper, Wen-Chin Huang, Tomoki Toda, and Junichi Yamagishi, “Generalization Ability of MOS Prediction Networks,” International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 8442–8446, 2022.

## Installation

### Clone and editable install
```bash
git clone https://github.com/fgnt/local_sqa.git
cd local_sqa
pip install -e .[fgnt]
```

### Via pip
```bash
pip install git+https://github.com/fgnt/local_sqa.git
```

## Data
We train our models on BVCC + NISQA.
Training can be extended to other datasets with utterance-level MOS annotations.

Please refer to the [data preparation instructions](local_sqa/data/README.md) for downloading and preparing the data.

## Models
We use SSL-based encoders with a simple decoder architecture (one BLSTM layer + one linear layer and average pooling).
Encoder configurations are provided at [conf/encoder](https://github.com/fgnt/local_sqa/tree/main/conf/encoder).

## Training
Training can be started with
```bash
python -m local_sqa.train
```
The default behaviour is as follows:
- Use `conf/default.yaml` as configuration
- Create a new directory under `./exp/` to save logs and checkpoints
- Load `bvcc.json` and `nisqa.json` from `local_sqa/data`
- Use `wav2vec2_base` as encoder

### Customization
We use [Hydra](https://hydra.cc/) for configuration management.

#### Changing where to save logs and checkpoints
You can change the output directory by overwriting `base_dir`.

#### Adding or removing a database
Databases are configured under the key `databases`.
You can add or remove databases by adding or deleting entries there.
To change the path pointing to the database structure file, overwrite `json_path`, e.g., `databases.bvcc.json_path=path/to/bvcc.json`.

## Evaluation
Instructions following soon.
