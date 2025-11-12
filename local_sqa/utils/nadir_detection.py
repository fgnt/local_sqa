import typing as tp

import numpy as np
import paderbox as pb
import scipy.signal


def conv_smoothing(
    signal, window_length=7, shift=1, threshold=3, end="cut",
    target=None, assert_shape: bool = True, reduce="sum",
):
    """

    Boundary effects are visible at beginning and end of signal.

    Examples:
        >>> conv_smoothing(np.array([False, True, True, True, False, False, False, True]), 3, 2)
        array([False,  True,  True,  True, False, False, False, False])

    Args:
        signal:
        window_length:
        threshold:

    Returns:

    """
    left_context = right_context = (window_length - 1) // 2
    if window_length % 2 == 0:
        right_context += 1
    act_conv = pb.array.segment_axis(
        np.pad(signal, (left_context, right_context), mode='constant'),
        length=window_length, shift=shift, axis=0, end=end,
    )
    if reduce == "sum":
        act_conv = np.sum(act_conv, axis=-1)
        act = act_conv >= threshold
    elif reduce == "mean":
        act = np.mean(act_conv, axis=-1)
    elif reduce == "max":
        act = np.max(act_conv, axis=-1)
    else:
        raise ValueError(f"Unknown reduction: {reduce}")
    if assert_shape:
        if target is None:
            target = signal
        assert act.shape == target.shape, (act.shape, target.shape)
    return act


def nadir_detection(
    scores: np.ndarray,
    sequence_length: tp.Optional[int] = None,
    height: tp.Optional[float] = None,
    width: tp.Optional[float] = None,
    prominence: tp.Optional[float] = None,
    rel_height: float = 0.5,
    wlen: tp.Optional[int] = None,
    normalize: bool = True,
):
    """Detect local minima in `scores` using scipy's `find_peaks`.

    Apply `find_peaks` on the inverted scores to find local minima.

    Args:
        scores (np.ndarray): 1d array of frame-level scores.
        sequence_length (int, optional): Length of the unpadded part of
            `scores`.
        height (float, optional): Minimum height of peaks for peak
            detection.
        width (float, optional): Minimum width of peaks for peak
            detection.
        prominence (float, optional): Minimum prominence of peaks for peak
            detection.
        rel_height (float, optional): Relative height of peaks for peak
            width estimation. Defaults to 0.5.
        wlen (int, optional): Window length for prominence and width
            estimation. Defaults to None.
        normalize (bool, optional): Whether to normalize scores to [0, 1].
            Defaults to True.

    Returns:
        prominence_array (np.ndarray): Array of same shape as scores containing
            the prominence of the detected local nadirs at each frame (0 where 
            no nadir is detected).
        (nadirs, properties): Output of `scipy.signal.find_peaks`.
    """
    scores = scores[:sequence_length]
    if normalize:
        scores = (scores - 1) / 4
        inv_scores = 1 - scores
    else:
        inv_scores = 5 - scores
    # Find local minima
    nadirs, properties = scipy.signal.find_peaks(
        inv_scores,
        height=height,
        width=width,
        prominence=prominence,
        rel_height=rel_height,
        wlen=wlen,
    )
    prominence_data = scipy.signal.peak_prominences(
        inv_scores, nadirs, wlen=wlen
    )
    _, _, left_ips, right_ips = scipy.signal.peak_widths(
        inv_scores, nadirs, prominence_data=prominence_data,
        rel_height=rel_height,
        wlen=wlen,
    )
    prominences, left_bases, right_bases = prominence_data
    properties.update({
        "left_ips": left_ips,
        "right_ips": right_ips,
        "prominences": prominences,
        "left_bases": left_bases,
        "right_bases": right_bases,
    })
    if len(prominences) > 0:
        prominence_arrays = []
        left_ips = np.floor(left_ips).astype(int)
        right_ips = np.ceil(right_ips).astype(int)
        for prom, left, right in zip(
            prominences,
            left_ips, right_ips,
        ):
            prominence_array = np.zeros_like(scores)
            prominence_array[left:right] = prom
            prominence_arrays.append(prominence_array)
        prominence_array = np.stack(prominence_arrays).max(axis=0)
    else:
        prominence_array = np.zeros_like(scores)
    return prominence_array, (nadirs, properties)
