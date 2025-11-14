import os
from pathlib import Path

import click
from paderbox.io import dump_json
from paderbox.io.audioread import audio_length, getparams
import pandas as pd
from tqdm import tqdm

DATABASE_ROOT = Path(os.environ.get('DATABASE_ROOT', ''))


def create_sim_subset(database_path: Path, split: str = "train"):
    split = split.upper()
    subdir = database_path / f"NISQA_{split}_SIM"
    df = pd.read_csv(subdir / f"NISQA_{split}_SIM_file.csv", sep=',')
    examples = {}
    for line in tqdm(df.itertuples(), total=len(df), desc=split):
        filename_deg = line.filename_deg
        filename_ref = line.filename_ref
        example_id = str(Path(filename_deg).stem)
        audio_path = str(subdir / "deg" / filename_deg)
        examples[example_id] = {
            'audio_path': {
                'observation': audio_path,
                'reference': str(subdir / "ref" / filename_ref),
            },
            'num_samples': audio_length(
                str(audio_path), unit='samples'
            ),
            'sampling_rate': getparams(str(audio_path)).framerate,
            'lang': line.lang,
            'votes': line.votes,
            'rating': {'mean': line.mos},
            'filter': line.filter,
            'bp_low': line.bp_low,
            'bp_high': line.bp_high,
            'codec1': line.codec1,
            'codec2': line.codec2,
            'codec3': line.codec3,
        }
    return examples


def create_json(database_path: Path):
    datasets = {}
    for split in "train val".split():
        datasets[f"{split}_sim"] = create_sim_subset(database_path, split=split)

    db = {'datasets': datasets}
    return db


@click.command()
@click.option(
    '--json-path', '-j',
    default='nisqa.json',
    help=(
        'Output path for the generated JSON file. If the '
        'file exists, it gets overwritten. Defaults to '
        '"nisqa.json".'
    ),
    type=click.Path(dir_okay=False, writable=True),
)
@click.option(
    '--database-path', '-db',
    default=DATABASE_ROOT / 'NISQA_Corpus',
    help='Path where the database is located.',
    type=click.Path(),
)
def main(
    json_path,
    database_path,
):
    database = create_json(
        Path(database_path).absolute(),
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
