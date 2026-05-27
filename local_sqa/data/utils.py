import numpy as np

from local_sqa.utils.audio_normalization import normalize_loudness


def prepare_audio(
    audio: np.ndarray,
    sampling_rate: int = 16_000,
    equal_loudness: bool = True,
    standardize_audio: bool = True,
):
    if equal_loudness:
        audio = normalize_loudness(
            audio, sampling_rate=sampling_rate
        )
    if standardize_audio:
        audio = (
            (audio - np.mean(audio, axis=-1, keepdims=True))
            / (np.std(audio, axis=-1, keepdims=True) + 1e-7)
        )
    return audio


def prepare_audio_example(
    example: dict,
    audio_key: str = "audio",
    equal_loudness: bool = True,
    standardize_audio: bool = True,
):
    sampling_rate = example["sampling_rate"]
    example[audio_key] = prepare_audio(
        example[audio_key],
        sampling_rate=sampling_rate,
        equal_loudness=equal_loudness,
        standardize_audio=standardize_audio,
    )
    return example
