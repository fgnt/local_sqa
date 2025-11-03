from functools import partial

import padertorch as pt
from padertorch.contrib.mk.typing import TSeqLen
from padertorch.ops.sequence.mask import compute_mask
import torch
from torch import Tensor


class Slicer(pt.Module):
    def __init__(
        self,
        sampling_rate: int,
        min_length_in_seconds: float = 1.0,
        max_length_in_seconds: float = 2.0,
        sequence_axis: int = -2,
    ):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.min_length_in_seconds = min_length_in_seconds
        self.max_length_in_seconds = max_length_in_seconds
        self.sequence_axis = sequence_axis

    def forward(
        self,
        x: Tensor,
        sequence_lengths: TSeqLen,
        rng=None,
    ):
        if sequence_lengths is None:
            raise ValueError("sequence_lengths must be provided")

        bs = x.shape[0]
        lengths = torch.randint(
            int(self.min_length_in_seconds * self.sampling_rate),
            int(self.max_length_in_seconds * self.sampling_rate),
            (bs,),
            generator=rng,
        ).long()
        lengths = torch.minimum(lengths, torch.tensor(sequence_lengths)-1)
        max_length = torch.amax(lengths).item()
        stops = torch.cat(list(map(
            partial(torch.randint, size=(1,), generator=rng),
            [max_length]*bs, sequence_lengths,
        ))).long()
        starts = stops - max_length
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
        return x, lengths, indices
