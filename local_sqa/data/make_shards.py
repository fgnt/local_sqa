import functools
from pathlib import Path
import typing as tp

from joblib import Parallel, delayed
from lazy_dataset.database import JsonDatabase
import numpy as np
from paderbox.array.interval import ArrayInterval
from paderbox.io import load_audio
from paderbox.transform.module_resample import resample
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from local_sqa.data.utils import prepare_audio


def prepare_audio_example(
    example,
    target_sampling_rate: int = 16_000,
    equal_loudness: bool = True,
    standardize_audio: bool = True,
    scores: tp.Optional[dict] = None,
    vad: tp.Optional["rVADfast"] = None,
    vad_transpose: tp.Optional[tp.Callable] = None,
):
    example["audio_data"] = {}
    example["vad"] = {}
    example["data_size"] = 0
    for k, p in example["audio_path"].items():
        audio, sr = load_audio(p, return_sample_rate=True, dtype='float32')
        if sr != target_sampling_rate:
            audio = resample(audio, in_rate=sr, out_rate=target_sampling_rate)
            example["sampling_rate"] = target_sampling_rate
            example["num_samples"] = audio.shape[-1]
        audio = prepare_audio(
            audio,
            sampling_rate=target_sampling_rate,
            equal_loudness=equal_loudness,
            standardize_audio=standardize_audio,
        )
        data = audio.tobytes()
        example["audio_data"][k] = data
        example["data_size"] += len(data)
        if scores is not None and k in scores:
            example_id = str(Path(p).stem)
            if example_id in scores[k].keys():
                if "scores" not in example:
                    example["scores"] = {}
                example_scores = scores[k][example_id]
                example["scores"][k] = {
                    k: list(v.values()) for k, v
                    in example_scores.to_dict().items()
                }
        if vad is not None:
            vad_mask, _ = vad(audio, target_sampling_rate)
            stride = int(vad.shift_duration * target_sampling_rate)
            win_len = int(vad.window_duration * target_sampling_rate)
            if vad_transpose is not None:
                vad_mask = vad_transpose(
                    vad_mask, np.ones(win_len), stride=stride,
                )[:audio.shape[-1]]
            ai = ArrayInterval(vad_mask.astype(bool))
            example["vad"][k] = list(ai.intervals)
    return example


def write_parquet_shards(
    json_file: str,
    dataset_names: list,
    out_dir: tp.Union[str, Path],
    *,
    target_sampling_rate: int = 16_000,
    equal_loudness: bool = True,
    standardize_audio: bool = True,
    apply_vad: bool = False,
    scores_roots: str = "",
    shard_size: int = 256 * 1024**2,
    row_group_size: int = 256,
    compression: str = "snappy",
    n_jobs: int = 8,
    seed: int = 0,
    dry_run: bool = False,
):
    """Write shards of parquet files for the given datasets.

    Args:
        json_file: Path to the JSON file containing the dataset.
        dataset_names: List of dataset names to process. If empty, all datasets
            in the JSON file will be processed.
        out_dir: Directory to write the output shards.
        target_sampling_rate: Sampling rate to resample the audio to. Defaults
            to 16 kHz.
        equal_loudness: Whether to apply equal loudness normalization. Defaults
            to True.
        standardize_audio: Whether to standardize the audio to zero mean and
            unit variance. Defaults to True.
        apply_vad: Whether to apply voice activity detection and store the
            resulting intervals. Defaults to False.
        scores_roots: Double double-colon ("::") separated list of
            channel_key:score_root pairs to include scores in the output. The
            score_root should be parsable by sed_scores_eval.base_modules.io.parse_scores.
        shard_size: Target size of each shard in bytes. Defaults to 256 MiB.
        row_group_size: Number of rows per row group in the parquet file.
            Defaults to 256.
        compression: Compression algorithm to use for the parquet files.
            Defaults to "snappy".
        n_jobs: Number of parallel jobs to use for processing the examples.
            Defaults to 8.
        seed: Random seed for shuffling the dataset. Defaults to 0.
        dry_run: If True, do not write any files. Defaults to False.
    """
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    db = JsonDatabase(json_file)

    if len(scores_roots) > 0:
        from sed_scores_eval.base_modules.io import parse_scores
        scores_roots = scores_roots.split("::")
        scores = {}
        for root in scores_roots:
            channel_key, score_root = root.split(":")
            scores_ds, _= parse_scores(score_root)
            scores[channel_key] = scores_ds
    else:
        scores = None

    rng = np.random.default_rng(seed)

    if apply_vad:
        from rVADfast import rVADfast
        import torch
        import torch.nn.functional as F
        vad = rVADfast(window_duration=0.025)
        vad_transpose = lambda inputs, weight, **kwargs: F.conv_transpose1d(
            torch.from_numpy(inputs[None]).double(),
            torch.from_numpy(weight[None, None]),
            **kwargs
        ).squeeze(0).numpy() > 0.5
    else:
        vad = None
        vad_transpose = None

    if len(dataset_names) == 0:
        dataset_names = db.dataset_names
    for dataset_name in dataset_names:
        ds = db.get_dataset(dataset_name)

        ds_dir = out_dir / dataset_name
        ds_dir.mkdir(parents=True, exist_ok=True)
        ds = ds.shuffle(rng=rng)
        shard_idx = 0
        batch = []
        batch_size = 0
        iterator = Parallel(
            n_jobs=n_jobs, return_as="generator_unordered",
            pre_dispatch="all", prefer="threads",
            batch_size="auto",
        )(
            delayed(functools.partial(
                prepare_audio_example,
                target_sampling_rate=target_sampling_rate,
                equal_loudness=equal_loudness,
                standardize_audio=standardize_audio,
                scores=scores,
                vad=vad,
                vad_transpose=vad_transpose,
            ))(example) for example in tqdm(
                ds, desc=f"Writing shards for {dataset_name} (pre-dispatch)",
            )
        )
        for example in tqdm(
            iterator, desc=f"Writing shards for {dataset_name}"
        ):
            out_path = ds_dir / f"{shard_idx:05d}.parquet"

            batch.append(example)
            batch_size += example["data_size"]

            if batch_size >= shard_size:
                table = pa.Table.from_pylist(batch)

                if not dry_run:
                    pq.write_table(
                        table,
                        out_path,
                        compression=compression,
                        row_group_size=row_group_size,
                        use_dictionary=True,
                        write_statistics=False,
                    )

                shard_idx += 1
                batch = []
                batch_size = 0

        if len(batch) > 0 and not dry_run:
            out_path = ds_dir / f"{shard_idx:05d}.parquet"
            table = pa.Table.from_pylist(batch)

            pq.write_table(
                table,
                out_path,
                compression=compression,
                row_group_size=row_group_size,
                use_dictionary=True,
                write_statistics=False,
            )


if __name__ == "__main__":
    import fire
    fire.Fire(write_parquet_shards)
