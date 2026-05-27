from functools import partial
import typing as tp

import numpy as np
import padertorch as pt
from padertorch.contrib.mk.typing import TSeqLen
from padertorch.ops.sequence.mask import compute_mask
from padertorch.data import example_to_device
import torch
from torch import Tensor


class Slicer(pt.Module):
    def __init__(
        self,
        sampling_rate: int,
        min_length_in_seconds: float = 1.0,
        max_length_in_seconds: float = 2.0,
        min_length_ratio: tp.Optional[float] = None,
        max_length_ratio: tp.Optional[float] = None,
        anchor: str = 'random',
        sequence_axis: int = 1,
        detach: bool = False,
    ):
        super().__init__()
        if anchor not in ['left', 'random']:
            raise ValueError("anchor must be one of ['left', 'random']")
        self.sampling_rate = sampling_rate
        self.min_length_in_seconds = min_length_in_seconds
        self.max_length_in_seconds = max_length_in_seconds
        self.min_length_ratio = min_length_ratio
        self.max_length_ratio = max_length_ratio
        self.anchor = anchor
        self.sequence_axis = sequence_axis
        self.detach = detach

    def reshape(
        self,
        targets: Tensor,
        slices: Tensor,
        indices: Tensor,
        seq_len_slices: TSeqLen,
    ) -> tp.Tuple[Tensor, Tensor, TSeqLen]:
        # Reshape targets to match sliced inputs
        indices = indices[..., 0]  # Remove feature dimension
        while indices.ndim < targets.ndim:
            indices = indices.unsqueeze(-1)
        indices = indices.expand(slices.shape)
        targets = targets.gather(1, indices)
        return targets, slices, seq_len_slices

    def forward(
        self,
        x: Tensor,
        sequence_lengths: TSeqLen,
        audio_start_samples: tp.Optional[tp.Union[np.ndarray, tp.List[int]]] = None,
        audio_stop_samples: tp.Optional[tp.Union[np.ndarray, tp.List[int]]] = None,
        rng=None,
    ):
        if sequence_lengths is None:
            raise ValueError("sequence_lengths must be provided")

        bs = x.shape[0]
        if (
            self.min_length_ratio is not None
            or self.max_length_ratio is not None
        ):
            lengths = []
            for seq_len in sequence_lengths:
                min_length = int(
                    self.min_length_ratio * seq_len
                ) if self.min_length_ratio is not None else int(
                    self.min_length_in_seconds * self.sampling_rate
                )
                max_length = int(
                    self.max_length_ratio * seq_len
                ) if self.max_length_ratio is not None else int(
                    self.max_length_in_seconds * self.sampling_rate
                )
                length = torch.randint(
                    min_length,
                    max(max_length, min_length) + 1,
                    (1,),
                    generator=rng,
                )
                lengths.append(length)
            lengths = torch.cat(lengths).long()
        else:
            lengths = torch.randint(
                int(self.min_length_in_seconds * self.sampling_rate),
                int(self.max_length_in_seconds * self.sampling_rate),
                (bs,),
                generator=rng,
            ).long()
        lengths = torch.minimum(lengths, torch.tensor(sequence_lengths)-1)
        max_length = torch.amax(lengths).item()
        if self.anchor == 'left':
            starts = torch.zeros((bs,), dtype=torch.long)
        else:  # random
            stops = torch.cat(list(map(
                partial(torch.randint, size=(1,), generator=rng),
                lengths, sequence_lengths,
            ))).long()
            if audio_stop_samples is not None:
                audio_stop_samples = example_to_device(
                    audio_stop_samples, stops.device
                )
                if isinstance(audio_stop_samples, list):
                    audio_stop_samples = torch.stack(audio_stop_samples)
                stops = torch.minimum(stops, audio_stop_samples)
            starts = torch.maximum(
                stops - max_length,
                torch.zeros((bs,), dtype=torch.long).to(stops.device)
            )
        if audio_start_samples is not None:
            audio_start_samples = example_to_device(
                audio_start_samples, starts.device
            )
            if isinstance(audio_start_samples, list):
                audio_start_samples = torch.stack(audio_start_samples)
            starts = torch.maximum(
                starts, audio_start_samples
            )
        indices = (
            torch.arange(max_length).unsqueeze(0) + starts.unsqueeze(1)
        )
        for _ in range(x.dim() - 2):
            indices = indices.unsqueeze(-1)
        indices = indices.moveaxis(1, self.sequence_axis).to(x.device)
        shape = list(x.shape)
        shape[self.sequence_axis] = max_length
        indices = indices.expand(*shape)
        x = x.gather(self.sequence_axis, indices)
        # Padding mask
        mask = compute_mask(
            x, lengths.to(x.device),
            batch_axis=0, sequence_axis=self.sequence_axis,
        )
        x = x * mask
        if self.detach:
            x = x.detach()
        return x, lengths, indices
