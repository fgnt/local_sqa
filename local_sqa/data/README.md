# Data preparation

## Download the databases

### BVCC

Download the BVCC dataset from [Zenodo](https://zenodo.org/records/6572573#.Yphw5y8RprQ) and follow the instructions in the accompanying README to extract and prepare the files.
It suffices to download only the `main` subset for training.

### NISQA

Follow the instructions from the [NISQA repository](https://github.com/gabrielmittag/NISQA/wiki/NISQA-Corpus) to download and prepare the NISQA corpus.

## Create JSON files

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

## Adding more databases

You can extend training to other databases by providing JSON files in the same format as above.
To use them for training, add a new entry under the `databases` key in the [configuration file](../conf/default.yaml).
