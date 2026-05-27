import os
from pathlib import Path

import click
from joblib import Parallel, delayed
import numpy as np
from paderbox.io import dump_json
from paderbox.io.audioread import audio_length, getparams
import pandas as pd
import re
from tqdm import tqdm

DATABASE_ROOT = Path(os.environ.get('DATABASE_ROOT', ''))
CONDITIONS = [
    "filter",
    "wbgn",
    "p50mnru",
    "bgn",
    "clipping",
    "arb_filter",
    "asl_in",
    "asl_out",
    "codec1",
    "codec2",
    "codec3",
    "plcMode1",
    "plcMode2",
    "plcMode3",
    "wbgn_snr",
    "bgn_snr",
    "tc_fer",
    "tc_nburst",
    "cl_th",
    "bp_low",
    "bp_high",
    "p50_q",
    "bMode1",
    "bMode2",
    "bMode3",
    "FER1",
    "FER2",
    "FER3",
    "asl_in_level",
    "asl_out_level",
]
DESC_TO_CON = {
    "Amp clipping": {"clipping": "x"},
    "BP": {"filter": "bandpass"},
    "White Noise": lambda snr: {"wbgn": "x", "wbgn_snr": float(snr)},
    "P50MNRU": lambda snr: {"p50mnru": "x", "p50_q": float(snr)},
    "Level": lambda level: {"asl_out": "x", "asl_out_level": float(level)},
    "BGN": lambda snr: {"bgn": "x", "bgn_snr": float(snr)},
    "EVS": lambda codec_idx: {f"codec{codec_idx}": "evs"},
    "Opus": lambda codec_idx: {f"codec{codec_idx}": "opus"},
    "AMR-NB": lambda codec_idx: {f"codec{codec_idx}": "amrnb"},
    "AMR-WB": lambda codec_idx: {f"codec{codec_idx}": "amrwb"},
    "G.711": lambda codec_idx: {f"codec{codec_idx}": "g711"},
    "G.722": lambda codec_idx: {f"codec{codec_idx}": "g722"},
}
CON_TO_DESC = {
    "clipping": "Amp clipping",
    "lowpass": "LPF",
    "highpass": "HPF",
    "bandpass": "BP",
    "wbgn": "White Noise",
    "p50mnru": "P50MNRU",
    "asl_out": "Level",
    "bgn": "BGN",
    "evs": "EVS",
    "opus": "Opus",
    "amrnb": "AMR-NB",
    "amrwb": "AMR-WB",
    "g711": "G.711",
    "g722": "G.722",
    "tc_fer": "FER",
    "FER1": "PL",
    "FER2": "PL",
    "FER3": "PL",
    "arb_filter": "Arbitrary Filter",
}


def parse_test_conditions(description: str):
    if description.startswith(
        ("WhatsApp", "Zoom", "Discord", "Skype", "VoLTE")
    ):
        # Live conditions
        return {"class": [description]}
    conditions = {"class": set()}
    cons = re.split(r"\s+\+\s+", description)
    codec_idx = 1
    increase_codec_idx = False
    for con in cons:
        con = con.strip()
        if con.startswith("Anchor"):
            _, con = con.split(" ", 1)
            conditions["anchor"] = "x"
        subcons = con.split("x")
        for subcon in subcons:
            if len(subcon) <= 1:
                continue
            subcon = subcon.strip()
            if "@" in subcon or subcon.startswith(("G", "AMR")):
                codec, *mode = subcon.split("@")
                codec, *_ = codec.split(" ")
                conditions["class"].update([codec.strip()])
                conditions.update({
                    **DESC_TO_CON[codec.strip()](codec_idx)
                })
                subcon = ""
                if len(mode) > 0:
                    mode = " ".join(mode)
                    mode = mode.strip()
                    mode, *other = mode.split(" ")
                    try:
                        conditions[f"bMode{codec_idx}"] = float(mode)
                    except ValueError:
                        mode = mode.rstrip("PL")
                        conditions[f"bMode{codec_idx}"] = float(mode)
                        other.insert(0, "PL")
                    if len(other) > 0:
                        subcon = " ".join(other)
                increase_codec_idx = True
            if subcon.startswith("BP"):
                class_str, bp_range = subcon.split(" ")
                bp_low, bp_high = bp_range.strip().split("-")
                conditions["class"].update([class_str.strip()])
                conditions.update({
                    **DESC_TO_CON[class_str],
                    'bp_low': float(bp_low.strip()),
                    'bp_high': float(bp_high.strip().rstrip("Hz")),
                })
            elif subcon.startswith("Time Clipping"):
                class_str, tc_fer = subcon.rsplit(" ", maxsplit=1)
                conditions["class"].update(["FER"])
                conditions.update({
                    "tc_fer": float(tc_fer.strip().rstrip("%")) / 100.0
                })
            elif subcon.startswith("Amp clipping"):
                class_str, cl_th = subcon.rsplit(" ", maxsplit=1)
                conditions["class"].update([class_str.strip()])
                conditions.update({
                    **DESC_TO_CON[class_str],
                    "cl_th": float(cl_th.strip()),
                })
            elif subcon.startswith("PL"):
                class_str, fer, *plc_mode = subcon.split(" ", maxsplit=2)
                conditions["class"].update([class_str.strip()])
                conditions.update({
                    f"FER{codec_idx}": (
                        float(fer.strip().rstrip("%")) / 100.0
                    )
                })
                if len(plc_mode) > 0:
                    plc_mode = " ".join(plc_mode)
                    value = plc_mode.strip()
                    conditions[f"plcMode{codec_idx}"] = value
                    desc = value.title() + " PLC"
                    conditions["class"].update([desc])
            elif subcon.rsplit(" ", maxsplit=1)[0] in DESC_TO_CON:
                subcon, level_or_snr = subcon.rsplit(" ", maxsplit=1)
                conditions["class"].update([subcon.strip()])
                conditions.update({
                    **DESC_TO_CON[subcon.strip()](
                        level_or_snr.strip().rstrip("dB")
                    )
                })
            codec_idx += increase_codec_idx
            increase_codec_idx = False
    conditions["class"] = sorted(list(conditions["class"]))
    return conditions


def _worker(line: pd.Series, subdir: Path, split: str):
    filename_deg = line.filename_deg
    filename_ref = line.filename_ref
    example_id = str(Path(filename_deg).stem)
    audio_path = str(subdir / "deg" / filename_deg)
    example = {
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
        'conditions': {}
    }
    if split == "TEST":
        description = line.con_description
        if "Fullband clean" in description:
            return example_id, example
        example["conditions"] = parse_test_conditions(
            description,
        )
    else:
        example["conditions"]["class"] = []
        for condition in CONDITIONS:
            value = getattr(line, condition, "-")
            if not (
                value == "-"
                or (isinstance(value, float) and np.isnan(value))
            ):
                example['conditions'][condition] = value
                if (
                    condition in ["plcMode1", "plcMode2", "plcMode3"]
                    and value in ["random", "bursty"]
                ):
                    desc = value.title() + " PLC"
                else:
                    desc = (
                        CON_TO_DESC.get(condition, None)
                        or CON_TO_DESC.get(str(value), None)
                    )
                if desc is not None:
                    example["conditions"]["class"].append(desc)
        example["conditions"]["class"] = sorted(set(
            example["conditions"]["class"]
        ))
    return example_id, example


def create_subset(
    database_path: Path, split: str = "train", suffix: str = "sim",
    num_workers: int = 8,
):
    split = split.upper()
    suffix = suffix.upper()
    subdir = database_path / f"NISQA_{split}_{suffix}"
    df = pd.read_csv(subdir / f"NISQA_{split}_{suffix}_file.csv", sep=',')
    examples = {}
    classes = set()
    combinations = set()
    results = Parallel(n_jobs=num_workers)(
        delayed(_worker)(line, subdir, split) for line in tqdm(
            df.itertuples(), total=len(df), desc=f"{split}_{suffix}"
        )
    )
    for example_id, example in results:
        examples[example_id] = example
        classes.update(example["conditions"].get("class", []))
        _classes = example["conditions"].get("class", [])
        if len(_classes) > 0:
            combinations.update(["+".join(_classes)])
    meta = {
        "num_classes": len(classes),
        "num_combinations": len(combinations),
        "classes": sorted(classes),
        "combinations": sorted(combinations),
    }
    return examples, meta


def create_json(database_path: Path, num_workers: int = 8):
    datasets = {}
    meta = {}
    for suffix in "for p501".split():
        ds, meta_test = create_subset(
            database_path, split="test", suffix=suffix
        )
        datasets[f"test_{suffix}"] = ds
        meta[f"test_{suffix}"] = meta_test
    for split in "val train".split():
        ds, _meta = create_subset(
            database_path, split=split, num_workers=num_workers
        )
        datasets[f"{split}_sim"] = ds
        meta[f"{split}_sim"] = _meta

    db = {'datasets': datasets, 'meta': meta}
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
@click.option(
    '--num-workers', type=int, default=8,
)
def main(
    json_path,
    database_path,
    num_workers,
):
    database = create_json(
        Path(database_path).absolute(),
        num_workers=num_workers,
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
