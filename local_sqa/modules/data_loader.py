from dataclasses import dataclass
import functools
from pathlib import Path
import random
import typing as tp
import warnings

import lazy_dataset
from lazy_dataset import Dataset
import numpy as np
from paderbox.io import load_audio, load_json
from paderbox.transform.module_resample import resample_sox
from padertorch import Configurable
from padertorch.data.segment import segment_axis
from padertorch.data.utils import collate_fn
import psutil

from local_sqa.data.nisqa.augment import get_mixup_mask


@dataclass
class LoadAudio(Configurable):
    audio_path_keys: tp.Union[str, tp.List[str]] = "audio_path.observation"
    target_sampling_rate: tp.Optional[int] = None
    resample: bool = False

    def get_audio_path(self, example: dict):
        if isinstance(self.audio_path_keys, str):
            audio_path = example
            for key in self.audio_path_keys.split('.'):
                audio_path = audio_path[key]
            return audio_path

        for audio_path_key in self.audio_path_keys:
            audio_path = example
            try:
                for key in audio_path_key.split('.'):
                    audio_path = audio_path[key]
            except KeyError:
                continue
            return audio_path

        raise ValueError(
            f"None of the audio path keys {self.audio_path_keys} found."
        )

    def __call__(self, example: dict):
        audio_path = self.get_audio_path(example)
        example["audio_path"] = audio_path
        audio, sr = load_audio(
            audio_path, return_sample_rate=True, dtype="float32"
        )
        if (
            self.target_sampling_rate is not None
            and sr != self.target_sampling_rate
        ):
            if self.resample:
                audio = resample_sox(
                    audio, in_rate=sr, out_rate=self.target_sampling_rate
                )
                sr = self.target_sampling_rate
            else:
                raise ValueError(
                    f"Sample rate {sr}Hz does not match "
                    f"target sample rate {self.target_sampling_rate}Hz "
                    f"and resample is set to False."
                )
        example["audio"] = audio
        example["num_samples"] = audio.shape[-1]
        example["sampling_rate"] = sr
        return example


def sort_by_length(examples: tp.List[dict]):
    examples = sorted(
        examples, key=lambda x: x["num_samples"], reverse=True
    )
    return examples


class ParquetDataset(Dataset):
    def __init__(
        self, pq_file: "ParquetFile",
        column_names: tp.Optional[tp.Sequence[str]] = None,
    ):
        self.pq_file = pq_file
        self.column_names = column_names

    @property
    def indexable(self):
        return False

    def __iter__(self, with_key=False):
        if with_key:
            raise NotImplementedError()
        for batch in self.pq_file.iter_batches(columns=self.column_names):
            for i in range(batch.num_rows):
                yield {
                    key: batch[key][i].as_py()
                    for key in batch.column_names
                }


@dataclass
class ParquetParser(Configurable):
    input_path: tp.Union[str, Path]
    train_dataset_names: tp.Optional[tp.Union[tp.Sequence[str], str]]
    val_dataset_names: tp.Optional[tp.Union[tp.Sequence[str], str]] = ()

    json_path: tp.Optional[tp.Union[str, Path]] = None
    batch_size: int = 256
    column_names: tp.Optional[tp.Sequence[str]] = None
    audio_data_key: str = "audio_data.observation"
    additional_keys: tp.Sequence[str] = ()
    shuffle: bool = True
    reshuffle: bool = True
    disable_validation_reshuffle: bool = True
    seed: int = 0
    disable: bool = False

    def __post_init__(self):
        self.input_path = Path(self.input_path)

    def _get_dataset(
        self, dataset_names: tp.Union[tp.Sequence[str], str],
        train: bool = True,
    ):
        if self.disable:
            return lazy_dataset.core.DictDataset({})
        import pyarrow.parquet as pq
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]
        pq_files = [
            pq.ParquetFile(file) for name in dataset_names
            for file in (self.input_path / name).glob("*.parquet")
        ]
        if self.shuffle:
            if not train:
                random.seed(self.seed)
            random.shuffle(pq_files)
        return list(map(
            functools.partial(ParquetDataset, column_names=self.column_names),
            pq_files
        ))

    def _process_dataset(self, dataset, train: bool = True):
        if (
            self.shuffle and self.reshuffle
            and (train or not self.disable_validation_reshuffle)
        ):
            if train:
                rng = None
            else:
                rng = np.random.default_rng(self.seed)
            dataset = dataset.shuffle(
                reshuffle=True, buffer_size=self.batch_size,
                rng=rng,
            )
        dataset = (
            dataset
            .map(self.decode_audio)
        )
        return dataset

    @property
    def train_dataset(self):
        return list(map(self.process_dataset,
            self._get_dataset(self.train_dataset_names)
        ))

    @property
    def val_dataset(self):
        if self.val_dataset_names is None or len(self.val_dataset_names) == 0:
            return lazy_dataset.core.DictDataset({})
        return list(map(
            functools.partial(self.process_dataset, train=False),
            self._get_dataset(self.val_dataset_names, train=False)
        ))

    def decode_audio(self, example: dict):
        audio_data = example
        for key in self.audio_data_key.split('.'):
            audio_data = audio_data[key]
        if isinstance(audio_data, dict):
            example["audio"] = {
                key: np.frombuffer(ad, dtype=np.float32)
                for key, ad in audio_data.items()
            }
            return example
        audio = np.frombuffer(audio_data, dtype=np.float32)
        example["audio"] = audio
        return example

    def finalize_example(self, example: dict, train: bool = True):
        return {
            "example_id": example["example_id"],
            "audio": example["audio"],
            "num_samples": example["num_samples"],
            "sampling_rate": example["sampling_rate"],
            **{
                key: example[key] for key in self.additional_keys
            }
        }

    def process_dataset(self, dataset, train: bool = True):
        dataset = self._process_dataset(dataset, train=train)
        dataset = dataset.map(
            functools.partial(self.finalize_example, train=train)
        )
        return dataset


@dataclass
class NISQAMixupParser(ParquetParser):
    p_mixup: float = 0.0
    observation_channel: str = "observation"
    reference_channel: str = "reference"
    min_segments: int = 1
    max_segments: int = 3
    target_rate: tp.Optional[float] = None
    min_mixup_duration: float = 0.2
    max_mixup_duration: float = 1.0
    min_mos_ref: float = 1.0
    max_mos_obs: float = 5.0
    mixup_conditions: bool = False
    apply_vad: bool = False
    diff_threshold: float = 0.
    exclude_conditions: tp.Sequence[str] = ()
    multi_label: bool = False
    ignore_clean: bool = False
    ignore_index: int = -100

    def __post_init__(self):
        super().__post_init__()
        if self.mixup_conditions and self.json_path is None:
            raise ValueError("json_path must be provided for NISQAMixupParser.")
        if self.json_path is not None:
            self.meta = load_json(self.json_path)["meta"]
            self.class2id = {"clean": 0}

    def _get_dataset(
        self, dataset_names: tp.Union[tp.Sequence[str], str],
        train: bool = True,
    ):
        dataset = super()._get_dataset(dataset_names, train=train)
        if train and self.mixup_conditions:
            if isinstance(dataset_names, str):
                dataset_names = [dataset_names]
            for dataset_name in dataset_names:
                classes = self.meta[dataset_name][
                    "classes" if self.multi_label else "combinations"
                ]
                idx = len(self.class2id)
                for class_ in classes:
                    if class_ not in self.class2id:
                        self.class2id[class_] = idx
                        idx += 1
        return dataset

    def _process_dataset(self, dataset, train: bool = True):
        dataset = super()._process_dataset(dataset, train=train)
        if self.p_mixup > 0.0 or self.mixup_conditions:
            if train:
                seed = None
            else:
                seed = self.seed
            rng = np.random.default_rng(seed=seed)
            dataset = dataset.map(
                functools.partial(self.mixup, rng=rng)
            ).map(
                self.add_condition_labels
            )
        return dataset

    def mixup(self, example: dict, rng: np.random.Generator):

        observation = example["audio"][self.observation_channel]
        reference = example["audio"][self.reference_channel]

        if (
            rng.uniform() >= self.p_mixup
            or observation.shape[-1] != reference.shape[-1]
            or "scores" not in example
        ):
            example["audio"] = observation
            return example

        sampling_rate = example["sampling_rate"]

        conditions = example["conditions"].get("class", [])
        if self.exclude_conditions and any(
            condition in self.exclude_conditions for condition in conditions
        ):
            example["audio"] = observation
            return example

        # Load scores
        obs_scores = np.asarray(
            example["scores"][self.observation_channel]["MOS"]
        )
        ref_scores = np.asarray(
            example["scores"][self.reference_channel]["MOS"]
        )
        mean_ref = ref_scores.mean()
        mean_obs = obs_scores.mean()
        if (
            (mean_ref <= mean_obs and self.mixup_conditions)
            or mean_ref < self.min_mos_ref
            or mean_obs > self.max_mos_obs
        ):
            # If reference is too bad or observation is too good, skip mixup
            example["audio"] = observation
            return example

        if self.apply_vad:
            vad_intervals = example["vad"][self.observation_channel]
        else:
            vad_intervals = None
        mixup_mask = get_mixup_mask(
            observation,
            diff_threshold=self.diff_threshold,
            sampling_rate=sampling_rate,
            max_segments=self.max_segments,
            min_segments=self.min_segments,
            dilate_duration=self.min_mixup_duration,
            min_length=int(self.min_mixup_duration * sampling_rate),
            max_length=int(self.max_mixup_duration * sampling_rate),
            target_rate=self.target_rate,
            scores_obs=(
                example["scores"][self.observation_channel]
                if self.mixup_conditions else None
            ),
            scores_ref=(
                example["scores"][self.reference_channel]
                if self.mixup_conditions else None
            ),
            vad_intervals=vad_intervals,
            rng=rng,
        )
        if mixup_mask is None:
            example["audio"] = observation
            return example
        mixup_mask = mixup_mask.astype(np.float32)
        mixed = (
            reference * (1.0 - mixup_mask) + observation * mixup_mask
        )

        x = np.asarray(
            example["scores"][self.observation_channel]["onset"]
        )
        xp = np.linspace(
            0, observation.shape[-1] / sampling_rate, observation.shape[-1]
        )
        mixup_mask_scores = np.interp(x, xp, mixup_mask) >= 0.5
        mixed_scores = (
            ref_scores * (1.0 - mixup_mask_scores)
            + obs_scores * mixup_mask_scores
        )

        example = {
            **example,
            "audio": mixed,
            "num_samples": mixed.shape[-1],
            "sampling_rate": example["sampling_rate"],
            "rating": {"mean": float(mixed_scores.mean())},
            "frame_mos": mixed_scores,
            "mixup_mask_scores": mixup_mask_scores,
        }

        return example

    def add_condition_labels(self, example: dict):
        if self.mixup_conditions:
            conditions = example["conditions"].get("class", [])
            if self.exclude_conditions and any(
                condition in self.exclude_conditions for condition in conditions
            ):
                return example
            # Add frame-wise condition labels for contrastive learning
            clean_label = self.ignore_index if self.ignore_clean else "clean"
            mixup_mask_scores = example.get("mixup_mask_scores")
            if mixup_mask_scores is None:
                try:
                    obs_scores = np.asarray(
                        example["scores"][self.observation_channel]["MOS"]
                    )
                    ref_scores = np.asarray(
                        example["scores"][self.reference_channel]["MOS"]
                    )
                except KeyError:
                    warnings.warn(
                        "No frame-level scores for example "
                        f"{example['example_id']} found"
                    )
                    return example
                mean_ref = ref_scores.mean()
                mean_obs = obs_scores.mean()
                if (
                    mean_ref <= mean_obs
                    or mean_ref < self.min_mos_ref
                    or mean_obs > self.max_mos_obs
                ):
                    if mean_ref < self.min_mos_ref:
                        # If reference is too bad, don't, add condition labels
                        return example
                    mixup_mask_scores = np.zeros_like(obs_scores)
                else:
                    mixup_mask_scores = (
                        (ref_scores - obs_scores) > self.diff_threshold
                    )
            clean_label = self.class2id.get(clean_label, self.ignore_index)
            condition_labels = [
                [clean_label] if self.multi_label else clean_label
            ] * mixup_mask_scores.shape[0]
            if len(conditions) > 0:
                if not self.multi_label:
                    conditions = self.class2id.get(
                        "+".join(conditions), self.ignore_index
                    )
                    # conditions = "+".join(conditions)
                else:
                    conditions = [
                        self.class2id.get(condition, self.ignore_index)
                        for condition in conditions
                    ]
                for index in np.where(mixup_mask_scores >= 0.5)[0]:
                    condition_labels[index] = conditions
            example.update({
                "condition_labels": condition_labels,
            })
        return example

    def finalize_example(self, example: dict, train: bool = True):
        if isinstance(example["audio"], dict):
            example["audio"] = example["audio"][self.observation_channel]
        return {
            "example_id": example.get("example_id", None),
            "audio": example["audio"],
            "num_samples": example["num_samples"],
            "sampling_rate": example["sampling_rate"],
            "frame_mos": example.get("frame_mos", []),
            "condition_labels": example.get("condition_labels", []),
            **{
                key: example[key] for key in self.additional_keys
            }
        }


@dataclass
class BVCCParser(ParquetParser):
    frame_duration: float = 0.025
    hop_duration: float = 0.02
    pad: bool = True
    add_condition_labels: bool = True
    validation_labels: bool = False
    ignore_clean: bool = False
    ignore_system: bool = False
    ignore_index: int = -100
    multi_label: bool = False

    def __post_init__(self):
        super().__post_init__()
        if (
            self.add_condition_labels
            and not self.ignore_system
            and self.json_path is None
        ):
            raise ValueError(
                "json_path must be provided for BVCCParser when "
                "add_condition_labels=True and ignore_system=False."
            )
        if self.json_path is not None:
            self.meta = load_json(self.json_path)["meta"]
            self.class2id = {"clean": 0}

    def _get_dataset(
        self, dataset_names: tp.Union[tp.Sequence[str], str],
        train: bool = True,
    ):
        dataset = super()._get_dataset(dataset_names, train=train)
        if train and self.add_condition_labels:
            if isinstance(dataset_names, str):
                dataset_names = [dataset_names]
            for dataset_name in dataset_names:
                classes = self.meta[dataset_name]["systems"]
                idx = len(self.class2id)
                for class_ in classes:
                    if class_ not in self.class2id:
                        self.class2id[class_] = idx
                        idx += 1
        return dataset

    def finalize_example(self, example: dict, train: bool = True):
        if (
            not self.add_condition_labels
            or (not train and not self.validation_labels)
        ):
            example = super().finalize_example(example)
            return {
                **example,
                "frame_mos": [],
                "condition_labels": [],
            }
        sampling_rate = example["sampling_rate"]
        frame_len = int(self.frame_duration * sampling_rate)
        hop_len = int(self.hop_duration * sampling_rate)
        num_frames = segment_axis(
            example["audio"], frame_len, hop_len, axis=-1,
            end="pad" if self.pad else "cut",
        ).shape[0]

        is_bona_fide = example["is_bona_fide"]
        if is_bona_fide:
            label = self.ignore_index if self.ignore_clean else 0  # "clean"
        else:
            if not self.ignore_system:
                raise NotImplementedError(
                    "System labels not implemented for BVCCParser."
                )
            label = (
                example["system"] if not self.ignore_system
                else self.ignore_index
            )

        condition_labels = [
            [label] if self.multi_label else label
        ] * num_frames

        return {
            "example_id": example.get("example_id", None),
            "audio": example["audio"],
            "num_samples": example["num_samples"],
            "sampling_rate": example["sampling_rate"],
            "frame_mos": [],
            "condition_labels": condition_labels,
            **{
                key: example[key] for key in self.additional_keys
            }
        }


class Dataloader(Configurable):
    def __init__(
        self,
        batch_size: int,
        stage: str = "train",
        shuffle: bool = True,
        reshuffle: bool = True,
        shuffle_buffer_size: tp.Optional[int] = None,
        prefetch_buffer_size: tp.Optional[int] = None,
        num_workers: int = 1,
        max_total_size: tp.Optional[int] = None,
        max_padding_rate: tp.Optional[float] = None,
        p_silence_aug: float = 0.0,
        min_silence_duration: float = 0.2,
        max_silence_duration: float = 2.0,
        silence_db: float = -20.0,
        seed: tp.Optional[int] = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.stage = stage
        self.shuffle = shuffle
        self.reshuffle = reshuffle
        self.prefetch_buffer_size = prefetch_buffer_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.num_workers = num_workers
        if self.num_workers < 0:
            self.num_workers = len(psutil.Process().cpu_affinity())
        if self.prefetch_buffer_size is None:
            self.prefetch_buffer_size = max(
                self.batch_size, self.num_workers
            )
        if self.shuffle_buffer_size is None:
            self.shuffle_buffer_size = 2 * self.batch_size
        self.max_total_size = max_total_size
        self.max_padding_rate = max_padding_rate
        self.p_silence_aug = p_silence_aug
        self.min_silence_duration = min_silence_duration
        self.max_silence_duration = max_silence_duration
        self.silence_db = silence_db
        self.seed = seed

    def __call__(self, *parsers: ParquetParser):
        if self.seed is not None:
            rng = np.random.default_rng(self.seed)
        else:
            rng = np.random.default_rng()
        datasets = [
            ds for parser in parsers
            for ds in getattr(parser, f"{self.stage}_dataset")
        ]
        if self.stage == "val":
            random.seed(self.seed)
        if self.shuffle:
            random.shuffle(datasets)
        dataset = lazy_dataset.concatenate(datasets)
        if self.stage == "train":
            dataset = dataset.map(
                functools.partial(self.augment, rng=rng)
            )
        if self.shuffle:
            if dataset.indexable:
                dataset = dataset.shuffle(rng=rng)
            elif self.reshuffle:
                dataset = dataset.shuffle(
                    reshuffle=True,
                    buffer_size=self.shuffle_buffer_size,
                    rng=rng,
                )
        if self.num_workers > 0:
            if dataset.indexable:
                dataset = dataset.prefetch(
                    self.num_workers, self.prefetch_buffer_size
                )
            else:
                dataset = dataset.prefetch(1, self.prefetch_buffer_size)
        dataset = (
            dataset
            .apply(self.batch)
            .map(sort_by_length)
            .map(collate_fn)
            .map(self.maybe_map_labels)
        )
        return dataset

    def augment(self, example: dict, rng: np.random.Generator):
        if rng.uniform() < self.p_silence_aug:
            min_silence_length = int(
                self.min_silence_duration * example["sampling_rate"]
            )
            max_silence_length = int(
                self.max_silence_duration * example["sampling_rate"]
            )
            start_silence = rng.normal(size=rng.integers(
                min_silence_length, max_silence_length
            )).astype(np.float32) * 10 ** (self.silence_db / 20)
            stop_silence = rng.normal(size=rng.integers(
                min_silence_length, max_silence_length
            )).astype(np.float32) * 10 ** (self.silence_db / 20)
            example["audio"] = np.concatenate((
                start_silence, example["audio"], stop_silence,
            ), axis=-1)
            example["num_samples"] = example["audio"].shape[-1]
            example["audio_start_samples"] = start_silence.shape[-1]
            example["audio_stop_samples"] = (
                example["num_samples"] - stop_silence.shape[-1]
            )
        else:
            example["audio_start_samples"] = 0
            example["audio_stop_samples"] = example["num_samples"]
        return example

    def batch(self, dataset: Dataset):
        if self.max_total_size is None and self.max_padding_rate is None:
            return dataset.batch(self.batch_size, drop_last=False)
        return dataset.batch_dynamic_time_series_bucket(
            self.batch_size, len_key="num_samples",
            max_padding_rate=self.max_padding_rate,
            max_total_size=self.max_total_size,
            drop_incomplete=False,
            sort_key="num_samples",
        )

    def maybe_map_labels(self, example: dict):
        return example
