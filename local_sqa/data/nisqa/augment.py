import math
import typing as tp

import numpy as np
from paderbox.array.interval import ArrayInterval
from paderbox.array.kernel import ai_dilate
import torch
import torch.nn.functional as F


def get_mixup_mask(
    observation: np.ndarray,
    sampling_rate: int,
    max_segments: int,
    scores_obs: tp.Optional[tp.Dict[str, list]] = None,
    scores_ref: tp.Optional[tp.Dict[str, list]] = None,
    diff_threshold: tp.Optional[float] = None,
    min_segments: int = 1,
    dilate_duration: float = 0.,
    min_length: int = 0,
    max_length: tp.Optional[int] = None,
    target_rate: tp.Optional[float] = None,
    vad_intervals: tp.Optional[list] = None,
    rng: np.random.Generator = np.random.default_rng(),
):
    mask_transpose = lambda inputs, weight, **kwargs: F.conv_transpose1d(
        torch.from_numpy(inputs[None]).double(),
        torch.from_numpy(weight[None, None]),
        **kwargs
    ).squeeze(0).numpy() > 0.5

    if scores_obs is not None or scores_ref is not None:
        if scores_obs is None or scores_ref is None:
            raise ValueError("Both scores_obs and scores_ref must be provided")
        # Find valid regions for sampling based on diff threshold
        diffs = (
            (np.array(scores_ref["MOS"]) - np.array(scores_obs["MOS"]))
            > diff_threshold
        )
        frame_len = math.ceil(
            np.diff(scores_obs["onset"]).mean() * sampling_rate
        )
        diffs = mask_transpose(
            diffs, np.ones(frame_len), stride=frame_len,
        )[:observation.shape[-1]]
        ai = ArrayInterval(diffs)
        if len(ai.intervals) == 0:
            # No valid regions
            return None
        # Dilate array interval by dilate_duration to get longer segments
        kernel_size = int(dilate_duration * sampling_rate)
        if kernel_size % 2 == 0:
            kernel_size += 1
        ai = ai_dilate(ai, kernel_size=kernel_size)
        if all((
            stop - start < min_length for (start, stop) in ai.intervals
        )):
            # Not enough valid regions
            return None
    else:
        ai = ArrayInterval(np.ones_like(observation, dtype=bool))

    if vad_intervals is not None:
        # Apply VAD to exclude silent regions
        vad_ai = ArrayInterval.from_pairs(
            vad_intervals, shape=observation.shape[-1]
        )
        ai &= vad_ai
        if all((
            stop - start < min_length for (start, stop) in ai.intervals
        )):
            # Not enough valid regions
            return None

    interval_lengths = np.array([
        stop - start for (start, stop) in ai.intervals
    ])
    ai_intervals = np.array(ai.intervals)
    mixup_mask = np.zeros_like(observation).astype(bool)
    if max_length is None:
        max_length = observation.shape[-1]
    if target_rate is None:
        num_segments = rng.integers(min_segments, max_segments + 1)
    else:
        mean_segment_length = (min_length + max_length) / 2
        est_num_segments = int(
            target_rate * observation.shape[-1] / mean_segment_length
        )
        min_segments = max(min_segments, est_num_segments - 1)
        max_segments = min(
            max(min_segments, max_segments), est_num_segments + 1
        )
        num_segments = rng.integers(min_segments, max_segments + 1)
    for _ in range(num_segments):
        # Draw random segment from valid regions
        interval = rng.choice(ai_intervals[interval_lengths >= min_length])
        segment_length = rng.integers(
            min_length,
            min(max_length + 1, interval[1] - interval[0] + 1)
        )
        max_start = interval[1] - segment_length
        start = rng.integers(interval[0], max_start + 1)
        mixup_mask[start:start + segment_length] = True

    return mixup_mask
