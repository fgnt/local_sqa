from collections import defaultdict
import os
from pathlib import Path
import typing as tp

import click
from paderbox.io import dump_json
from paderbox.io.audioread import audio_length
from paderbox.utils.nested import nested_merge
from tqdm.auto import tqdm

DATABASE_ROOT = Path(os.environ.get('DATABASE_ROOT', ''))


def read_averaged_scores(dataset_path: Path):
    ood = "ood" in dataset_path.name
    if ood:
        suffix = "ood"
    else:
        suffix = "main"
    data_dir = dataset_path / "DATA"
    dataset = {}
    num_samples = {}
    # Map secret keys to example ids
    if not ood:
        # Transcriptions are only available for the main track
        secret_to_example_id = {}
        with open(
            dataset_path / "secret_utt_mappings.txt", "r"
        ) as fid:
            lines = fid.readlines()
            for line in lines:
                example_id, secret_key = map(
                    lambda key: str(Path(key).stem), line.strip().split()
                )
                secret_to_example_id[secret_key] = example_id
        # Read transcriptions
        transcripts = {}
        with open (
            dataset_path / "main_track_truth_transcripts.txt", "r"
        ) as fid:
            lines = fid.readlines()
            for line in lines:
                secret_key, transcript = line.strip().split("\t")
                example_id = secret_to_example_id[secret_key]
                transcripts[example_id] = transcript.strip()
    else:
        transcripts = None
    subsets = "val train test".split()
    if ood:
        subsets.append("unlabeled")
    for subset in subsets:
        filepath = data_dir / "sets" / f"{subset}_mos_list.txt"
        examples = {}
        with open(filepath, "r", encoding="utf-8") as fid:
            lines = fid.readlines()
        for line in tqdm(lines, desc=f"{subset} {suffix} average"):
            entries = line.strip().split(",")
            try:
                filename, score = entries
            except ValueError:
                # Unlabeled set has no scores
                filename = entries[0]
                score = None
            system = filename.split("-")[0]
            example_id = str(Path(filename).stem)
            audio_path = data_dir / "wav" / f"{filename}"
            num_samples[example_id] = audio_length(
                str(audio_path), unit='samples'
            )
            examples[example_id] = {
                "example_id" : example_id,
                "audio_path" : {"observation": audio_path},
                "num_samples": num_samples[example_id],
                "system": system,
                "sampling_rate": 16_000,
            }
            if score is not None:
                examples[example_id].update({
                    "rating":  {"mean": float(score)}
                })
            if transcripts is not None:
                examples[example_id]["transcription"] = transcripts[example_id]
        dataset[f"{subset}_{suffix}"] = examples
    return dataset


def read_listener_scores(dataset_path: Path, average_scores: dict):
    ood = "ood" in dataset_path.name
    if ood:
        suffix = "ood"
    else:
        suffix = "main"
    data_dir = dataset_path / "DATA"
    dataset = {}
    subsets = "DEVSET TRAINSET TESTSET".split()
    if ood:
        subsets.append("UNLABELEDSET")
    for subset in subsets:
        filepath = data_dir / "sets" / subset
        examples = {}
        with open(filepath, "r", encoding="utf-8") as fid:
            lines = fid.readlines()
        subset = subset.split('SET')[0].lower()
        if subset == "dev":
            subset = "val"
        for line in tqdm(lines, desc=f"listener scores ({subset})"):
            try:
                _, filename, score, _, listener_info = line.strip()\
                    .split(",")
            except ValueError:
                continue
            listener_id = listener_info.split("_")[2]
            example_id = str(Path(filename).stem)
            if example_id not in examples:
                examples[example_id] = {
                    "example_id": example_id,
                    "listener_ratings": [],
                    "listener_ids": [],
                }
                try:
                    examples[example_id].update({
                        "rating": {
                            "mean": (
                                average_scores
                                [f"{subset}_{suffix}"]
                                [example_id]
                                ["rating"]
                                ["mean"]
                            )
                        },
                    })
                except KeyError:
                    # OOD unlabeled has no scores
                    pass
            examples[example_id]["listener_ratings"].append(float(score))
            examples[example_id]["listener_ids"].append(listener_id)
        dataset[f"{subset}_{suffix}"] = examples
    return dataset


def create_json(
    database_path,
    listener: bool = False,
    subsets: tp.Sequence = ("main",),
):
    database = {"datasets": {}, "alias": defaultdict(list)}
    for track in subsets:
        datasets = read_averaged_scores(database_path / track)
        if listener:
            listener_scores = read_listener_scores(
                database_path / track, datasets,
            )
            datasets = nested_merge(datasets, listener_scores)
        database["datasets"].update(datasets)
        database["alias"]["train"].append("train_" + track)
        for subset in "train val test".split():
            database["alias"][track].append(f"{subset}_{track}")
        if track == "ood":
            database["alias"]["ood"].append("unlabeled_ood")
    return database


@click.command()
@click.option(
    '--json-path', '-j',
    default='bvcc.json',
    help=(
        'Output path for the generated JSON file. If the '
        'file exists, it gets overwritten. Defaults to '
        '"bvcc.json".'
    ),
    type=click.Path(dir_okay=False, writable=True),
)
@click.option(
    '--database-path', '-db',
    default=DATABASE_ROOT / 'BVCC',
    help='Path where the database is located.',
    type=click.Path(),
)
@click.option(
    '--listener', is_flag=True,
    help=(
        'If set, the individual listener scores are included in the JSON file.'
    ),
)
@click.option(
    '--subset', '-s', type=click.Choice(["main", "ood"]),
    multiple=True, default='main',
    help=(
        'Subset(s) to include in the JSON file. '
        'Can be specified multiple times. '
        'Defaults to "main".'
    ),
)
def main(
    json_path,
    database_path,
    listener: bool = False,
    subsets: tp.Sequence = ("main",),
):
    database = create_json(
        Path(database_path).absolute(),
        listener, subsets,
    )
    dump_json(
        database,
        Path(json_path),
        create_path=True,
        indent=4,
        ensure_ascii=False,
    )


if __name__ == '__main__':
    main()
