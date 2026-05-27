# Localizable Speech Quality Assessment

Conventional speech quality assessment (SQA) models predict a single (global) score for the entire audio clip.
However, speech degradations not always persist through the entire utterance, but only partially.
A global score does not reflect *where* degradations happen.

This repository implements a framework for training SQA models that predict frame-level (quality) scores.
Building upon SSL-MOS [1], we use utterance-level quality labels and add various regularization losses to learn frame-level scores that adhere to local quality changes.
The following regularization losses are implemented:

- A consistency constraint that brings the encoder outputs of audio segments *within* and *detached* from the context close to each other in the embedding space [2].
- A frame-level loss with pseudo-labels generated from a partial mix-up data augmentation strategy [3].
- A contrastive loss to separate embeddings of different degradation types in the embedding space [3].

[1] Erica Cooper, Wen-Chin Huang, Tomoki Toda, and Junichi Yamagishi, “Generalization Ability of MOS Prediction Networks,” International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2022.  
[2] Michael Kuhlmann, Alexander Werning, Thilo von Neumann, and Reinhold Haeb-Umbach, "Speech Quality-based Localization of Low-Quality Speech and Text-to-Speech Synthesis Artifacts", International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2026.  
[3] Michael Kuhlmann, Tobias Cord-Landwehr, and Reinhold Haeb-Umbach, "Speech Quality Embeddings for Improved Detection and Classification of Degradations in Speech Signals", Odyssey Workshop, 2026.

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
We use SSL-based encoders with a simple decoder architecture.
Encoder and decoder configurations are provided at [conf/encoder](https://github.com/fgnt/local_sqa/tree/main/local_sqa/conf/encoder) and [conf/decoder](https://github.com/fgnt/local_sqa/tree/main/local_sqa/conf/decoder), respectively.

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
- Use the `conv` decoder
- Apply the consistency loss with $\lambda_\text{emb}=10$ and $\lambda_\text{scores}=1$

You can visualize the training progress with TensorBoard:
```bash
tensorboard --logdir ./exp/
```

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
