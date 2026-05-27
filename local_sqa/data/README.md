# Data preparation

This page guides you through the process of preparing the data for training SQA models.

## Step 1: Download the databases

### BVCC

Download the BVCC dataset from [Zenodo](https://zenodo.org/records/6572573#.Yphw5y8RprQ) and follow the instructions in the accompanying README to extract and prepare the files.
It suffices to download only the `main` subset for training.

### NISQA

Follow the instructions from the [NISQA repository](https://github.com/gabrielmittag/NISQA/wiki/NISQA-Corpus) to download and prepare the NISQA corpus.

## Step 2: Create JSON files

We use JSON files to provide a unified interface to different databases.
The structure is as follows (example for BVCC):

```json
{
  "datasets": {
    "test_main": {
      "sys00691-utt12c197c": {
        "audio_path": {
            "observation": ".../DATA/wav/sys00691-utt12c197c.wav"
        },
        "example_id": "sys00691-utt12c197c",
        "num_samples": 40253,
        "rating": {
            "mean": 4.0
        },
        "sampling_rate": 16000,
        "system": "sys00691"
      },
      ...
    },
    ...
  },
}
```

Different datasets are grouped under the `datasets` key.
Each dataset contains multiple examples identified by their `example_id`.
Each example contains metadata such as the path to the audio file, the number of samples, the mean rating, and the sampling rate.

To create the JSON files for BVCC and NISQA, run
```bash
python bvcc/create_json.py -db /path/to/bvcc
python nisqa/create_json.py -db /path/to/nisqa
```
This will create `bvcc.json` and `nisqa.json` in this directory, i.e., `local_sqa/data`.

### Adding more databases

You can extend training to other databases by providing JSON files in the same format as above.
To use them for training, add a new entry under the `databases` key in the [configuration file](../conf/default.yaml).

## Step 3: Create shards

To speed up training, we convert the audio files into shards of a fixed size.
This is done with the `make_shards.py` script.
The script takes the JSON files created in the previous step as input and creates shards in the specified output directory.
For example, to create shards for BVCC, run
```bash
python make_shards.py bvcc.json '[train_main]' /path/to/shards
```
This will create shards for the train-main split of BVCC in the specified output directory.
The script also supports various options for preprocessing the audio, such as resampling, equal loudness normalization, and voice activity detection.
For more details, run
```bash
python make_shards.py -h
```

### Preparing frame-level pseudo-labels for partial mix-up

To train with the frame-level loss, we add frame-level predictions from a pre-trained SQA model to the shards.
This allows us to perform the partial mix-up data augmentation on-the-fly during training.
To add the frame-level predictions, run
```bash
python make_shards.py \
  nisqa.json '[train_sim]' /path/to/shards \
  --scores_roots observation:/path/to/observation/scores::reference:/path/to/reference/scores
```
The `scores_roots` argument takes a double double-colon ("::") separated list of `channel_key:score_root` pairs.
The `channel_key` is the key under which the scores will be stored in the shard, and `score_root` should point to a directory of .tsv files containing the frame-level predictions for each example (as obtained by [infer.py](../infer.py)).
