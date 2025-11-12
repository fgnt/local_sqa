from cached_property import cached_property
from dataclasses import dataclass
from pathlib import Path
import typing as tp

import lazy_dataset
from lazy_dataset import Dataset
from lazy_dataset.database import JsonDatabase
import numpy as np
from paderbox.io import load_audio
from paderbox.transform.module_resample import resample_sox
from padertorch import Configurable
from padertorch.data.utils import collate_fn
import psutil


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
                    f"Sample rate {sr} does not match "
                    f"target sample rate {self.target_sampling_rate} "
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


@dataclass
class JsonParser(Configurable):
    json_path: tp.Union[str, Path]
    train_dataset_names: tp.Union[tp.Sequence[str], str]
    val_dataset_names: tp.Optional[tp.Union[tp.Sequence[str], str]] = ()

    audio_path_key: str = "audio_path.observation"
    mos_key: str = "mos"
    min_length_in_seconds: float = 0.0
    max_length_in_seconds: tp.Optional[float] = None

    def __post_init__(self):
        self.database = JsonDatabase(self.json_path)

    @cached_property
    def train_dataset(self):
        dataset = self.database.get_dataset(self.train_dataset_names)
        return self._process_dataset(dataset)

    @cached_property
    def val_dataset(self):
        if self.val_dataset_names is None or len(self.val_dataset_names) == 0:
            return lazy_dataset.core.DictDataset({})
        dataset = self.database.get_dataset(self.val_dataset_names)
        return self._process_dataset(dataset)

    def _process_dataset(self, dataset):
        return (
            dataset
            .filter(self.filter_min_length, lazy=False)
            .filter(self.filter_max_length, lazy=False)
            .map(LoadAudio(self.audio_path_key))
            .map(self.extract_mos_rating)
            .map(self.finalize_example)
        )

    def filter_min_length(self, example: dict):
        num_samples = example["num_samples"]
        min_length = int(
            self.min_length_in_seconds * example["sampling_rate"]
        )
        return num_samples >= min_length

    def filter_max_length(self, example: dict):
        if self.max_length_in_seconds is None:
            return True
        num_samples = example["num_samples"]
        max_length = int(
            self.max_length_in_seconds * example["sampling_rate"]
        )
        return num_samples <= max_length

    def extract_mos_rating(self, example: dict):
        mos = example
        for key in self.mos_key.split('.'):
            mos = mos[key]
        example["mos"] = mos
        return example

    def finalize_example(self, example: dict):
        return {
            "audio": example["audio"],
            "num_samples": example["num_samples"],
            "sampling_rate": example["sampling_rate"],
            "mos": example["mos"],
        }


class Dataloader(Configurable):
    def __init__(
        self,
        batch_size: int,
        stage: str = "train",
        target_sampling_rate: tp.Optional[int] = None,
        resample: bool = False,
        shuffle: bool = True,
        buffer_size: tp.Optional[int] = None,
        num_workers: int = 1,
        max_total_size: tp.Optional[int] = None,
        max_padding_rate: tp.Optional[float] = None,
        seed: tp.Optional[int] = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.stage = stage
        self.target_sampling_rate = target_sampling_rate
        self.resample = resample
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        if self.buffer_size is None:
            self.buffer_size = 2 * self.batch_size
        self.num_workers = num_workers
        if self.num_workers < 0:
            self.num_workers = len(psutil.Process().cpu_affinity())
        self.max_total_size = max_total_size
        self.max_padding_rate = max_padding_rate
        self.seed = seed

    def __call__(
        self, *json_parsers: JsonParser,
        prepare_example_fn: tp.Optional[tp.Callable] = None,
    ):
        if self.seed is not None:
            rng = np.random.default_rng(self.seed)
        else:
            rng = None
        datasets = [
            getattr(json_parser, f"{self.stage}_dataset")
            for json_parser in json_parsers
        ]
        dataset = lazy_dataset.concatenate(datasets)
        dataset = dataset.map(self.maybe_resample)
        if prepare_example_fn is not None:
            dataset = dataset.map(prepare_example_fn)
        if self.shuffle:
            dataset = dataset.shuffle(rng=rng)
        if self.num_workers > 0:
            dataset = dataset.prefetch(self.num_workers, self.buffer_size)
        dataset = (
            dataset
            .apply(self.batch)
            .map(sort_by_length)
            .map(collate_fn)
        )
        return dataset

    def maybe_resample(self, example: dict):
        sr = example["sampling_rate"]
        audio = example["audio"]
        if (
            self.target_sampling_rate is not None
            and sr != self.target_sampling_rate
        ):
            if self.resample:
                audio = resample_sox(
                    audio, in_rate=sr, out_rate=self.target_sampling_rate
                )
            else:
                raise ValueError(
                    f"Sample rate {sr} does not match "
                    f"target sample rate {self.target_sampling_rate} "
                    f"and resample is set to False."
                )
        example["audio"] = audio
        example["num_samples"] = audio.shape[-1]
        example["sampling_rate"] = self.target_sampling_rate or sr
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
