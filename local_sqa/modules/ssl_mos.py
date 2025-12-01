from collections import defaultdict
from functools import partial
from pathlib import Path
import typing as tp

import numpy as np
from paderbox.utils.nested import nested_merge
import padertorch as pt
from padertorch.contrib.je.modules.reduce import Mean
from padertorch.contrib.mk.typing import TSeqLen
from padertorch.data.segment import segment_axis
from padertorch.data.utils import collate_fn
from padertorch.ops.mappings import ACTIVATION_FN_MAP
from padertorch.utils import to_numpy
import scipy
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ..utils.audio_normalization import normalize_loudness

SAMPLING_RATE = 16_000


class SSLMOS(pt.Model):
    def __init__(
        self,
        encoder: pt.Module,
        criterion,
        proj_in_size: int,
        loss_weights=None,
        decoder: tp.Optional[pt.Module] = None,
        out_activation=None,
        input_key: str = "audio",
        input_seq_len_key: str = "num_samples",
        target_key: str = "rating",
        scale: float = 1.,
        bias: float = 0.,
        margin: float = 0.1,
        mae_clip: float = 0.1,
        normalize_ratings: bool = True,
        standardize_audio: bool = True,
        equal_loudness: bool = True,
        l2_normalization: bool = True,
        zero_init: bool = False,
        slicer: tp.Optional[pt.Module] = None,
        forget_gate_bias: tp.Optional[float] = None,
        take_last: bool = False,
    ):
        super().__init__()
        if margin < 0:
            raise ValueError(f"margin must be non-negative, got {margin}")

        self.criterion = criterion
        self.loss_weights = loss_weights
        if out_activation is not None:
            self.out_activation = ACTIVATION_FN_MAP[out_activation]()
        else:
            self.out_activation = nn.Identity()

        if isinstance(decoder, nn.LSTM) and forget_gate_bias is not None:
            for name, param in decoder.named_parameters():
                if "bias" in name:
                    # Initialize forget gate bias
                    # See: https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745/4
                    n = param.size(0)
                    param.data[n//4:n//2].fill_(forget_gate_bias)

        self.encoder = encoder
        self.decoder = decoder
        if proj_in_size > 1:
            self.out_proj = nn.Linear(proj_in_size, 1, bias=False)

        self.input_key = input_key
        self.input_seq_len_key = input_seq_len_key
        self.target_key = target_key
        self.scale = scale
        self.bias = bias
        self.margin = margin
        self.mae_clip = mae_clip
        self.normalize_ratings = normalize_ratings
        self.standardize_audio = standardize_audio
        self.equal_loudness = equal_loudness
        self.l2_normalization = l2_normalization
        self.slicer = slicer
        self.take_last = take_last

        self.reset_parameters()
        if zero_init:
            self.zero_init_()

    @classmethod
    def finalize_dogmatic_config(cls, config):
        config["criterion"] = {
            'factory': nn.L1Loss,
            'reduction': 'none',
        }

    def _normalize_ratings(self, ratings):
        return (ratings - 1) / 2 - 1  # [-1, 1]

    def zero_init_(self):
        if not hasattr(self, 'out_proj'):
            raise ValueError("Model has no out_proj layer to zero initialize.")
        for param in self.out_proj.parameters():
            param.detach().zero_()

    def inverse_normalization(self, scores: tp.Union[Tensor, np.ndarray]):
        if not self.normalize_ratings:
            return scores
        return (scores + 1) * 2 + 1  # [1, 5]

    def reset_parameters(self, seed=None):
        try:
            self.decoder.reset_parameters(seed=seed)
        except (AttributeError, TypeError):
            pass
        if not hasattr(self, 'out_proj'):
            return
        generator = torch.Generator(device=self.out_proj.weight.device)
        if seed is not None:
            generator.manual_seed(seed)
        nn.init.xavier_uniform_(self.out_proj.weight, generator=generator)
        # nn.init.zeros_(self.out_proj.bias)

    def example_to_device(self, example, device=None, memo=None):
        example = super().example_to_device(example, device, memo)
        audio = example[self.input_key]
        if isinstance(audio, list):
            audio = pt.pad_sequence(audio, batch_first=True)
            example[self.input_key] = audio
        return example

    def prepare_example(self, example):
        observation = example[self.input_key]
        if self.equal_loudness:
            if observation.ndim > 2:
                observation = np.stack(list(map(
                    partial(
                        normalize_loudness,
                        sampling_rate=example["sampling_rate"]
                    ), observation,
                )))
            else:
                observation = normalize_loudness(
                    observation, sampling_rate=example["sampling_rate"]
                )
        if self.standardize_audio:
            observation = (
                (observation - np.mean(observation, axis=-1, keepdims=True))
                / (np.std(observation, axis=-1, keepdims=True) + 1e-7)
            )
        example[self.input_key] = observation
        return example

    def test_seed(self, subiterator, seed, device=None):
        summaries = None
        self.eval()
        self.reset_parameters(seed)
        for batch in subiterator:
            batch = self.example_to_device(batch, device=device)
            with torch.no_grad():
                outputs = self.forward(batch)
                summary = self.review(batch, outputs)
            buffers = summary.pop("buffers")
            del summary["histograms"]
            if summaries is None:
                summaries = summary
                summaries["buffers"] = defaultdict(list)
                summaries["snapshots"] = {}
                summaries["histograms"] = {}
            else:
                summaries = nested_merge(summaries, summary)
            for key, value in buffers.items():
                summaries["buffers"][key].append(value)
            del summary

        summaries = self.modify_summary(summaries)
        return summaries

    def encode(
        self,
        wavs: tp.Optional[Tensor],
        num_samples: TSeqLen,
        *,
        latents: tp.Optional[Tensor] = None,
        seq_len_latents: TSeqLen = None,
    ):
        if latents is None:
            latents, seq_len_latents = self.encoder(
                wavs, num_samples, return_latents=True
            )
        x = self.encoder.extract_features_from_latents(
            latents, seq_len_latents
        )
        return latents, x, seq_len_latents

    def normalize_encoder_output(
        self, x: Tensor, seq_len_x: TSeqLen = None,
    ) -> Tensor:
        if self.l2_normalization:
            # Keeps losses low at initialization
            m = Mean(axis=1, keepdims=True)(x.detach(), seq_len_x)
            norm = torch.linalg.norm(m, ord=2, dim=-1, keepdim=True)
            x = x / (norm + 1e-6)
        return x

    def transform(
        self, x: Tensor, seq_len_x: TSeqLen = None, enforce_sorted=True
    ):
        if self.decoder is not None:
            if isinstance(self.decoder, nn.LSTM):
                if seq_len_x is not None:
                    x = pt.pack_padded_sequence(
                        x, seq_len_x, batch_first=True,
                        enforce_sorted=enforce_sorted,
                    )
                x, _ = self.decoder(x)
                if seq_len_x is not None:
                    x, _ = pt.pad_packed_sequence(x, batch_first=True)
            else:
                x, _ = self.decoder(x, seq_len_x)
        return x, seq_len_x

    def reduce(
        self, x: Tensor, seq_len_x: TSeqLen,
    ):
        return Mean(axis=1)(x, seq_len_x)

    def project(self, x: Tensor, seq_len_x: TSeqLen):
        if hasattr(self, 'out_proj'):
            x = self.out_proj(x)
        preds = self.out_activation(x).squeeze(-1)
        preds = preds*self.scale+self.bias
        if self.take_last:
            if seq_len_x is None:
                return preds[:, -1], preds
            seq_len_x = torch.tensor(seq_len_x, device=preds.device)
            if self.training:
                # Noisy last frame during training for better generalization
                # Shift by +/- 10% of sequence length
                rel_shift = (
                    torch.rand_like(seq_len_x.float(), device=preds.device)
                    * 0.2 - 0.1
                )
                shift = (rel_shift * seq_len_x.float()).long()
                seq_len_x = torch.minimum(seq_len_x + shift, seq_len_x.max())
            utt_preds = preds.gather(1, seq_len_x.unsqueeze(1)-1).squeeze(1)
            return utt_preds, preds
        return self.reduce(preds, seq_len_x), preds

    def finalize_summary(self, summary: dict):
        losses = summary.pop("losses")
        loss = 0.
        for key, value in losses.items():
            if self.loss_weights is None:
                weight = 1.
            else:
                weight = self.loss_weights[key]
            loss += weight * value
            summary["scalars"][key] = value.item()
            summary["scalars"][key + "_weight"] = weight
        summary["loss"] = loss
        return summary

    def forward(self, inputs: dict):
        wavs = inputs[self.input_key]
        sequence_lengths = inputs[self.input_seq_len_key]

        latents, embds, seq_len_embds = self.encode(
            wavs, sequence_lengths,
        )
        embds = self.normalize_encoder_output(embds, seq_len_embds)
        y, seq_len_y = self.transform(embds, seq_len_embds)  # Decoder embeddings
        preds, frame_preds = self.project(y, seq_len_y)
        return (
            frame_preds, seq_len_embds, preds, (latents, embds)
        )

    def review(self, inputs: dict, outputs: dict):
        summary = {'losses': {}, 'histograms': {}, 'buffers': {}, 'scalars': {}}
        summary["scalars"]["scale"] = self.scale
        summary["scalars"]["bias"] = self.bias

        # Prepare targets
        frame_preds, seq_len_y, preds, (latents, embds) = outputs
        ratings = inputs
        for k in self.target_key.split("."):
            ratings = ratings[k]
        ratings = torch.tensor(ratings)
        targets = ratings.to(preds.device)
        targets_mask = targets >= 1
        if self.normalize_ratings:
            targets = self._normalize_ratings(targets)
        if targets.ndim != 1:
            raise ValueError(
                f"Expected targets to have shape {preds.shape}, "
                f"got {targets.shape}"
            )
        if preds.ndim == 0:
            preds = preds.unsqueeze(0)
        assert preds.shape == targets.shape, (preds.shape, targets.shape)

        # Utterance-level regression loss + contrastive loss
        regr = self.criterion(preds, targets)
        diff = targets[:, None] - targets[None]
        diff_preds = preds[:, None] - preds[None]
        contrastive_loss = torch.triu(
            torch.relu((diff-diff_preds).abs()-self.margin)
        )
        regr_mask = (
            ((preds-targets).abs() > self.mae_clip)
            & targets_mask
        )
        contrast_mask = torch.triu(
            targets_mask.float()[:, None] * targets_mask.float()[None]
        )
        summary["losses"]["regression"] = (
            (regr*regr_mask).sum()/(regr_mask.sum()+1e-6)
        )
        summary["losses"]["contrastive_loss"] = (
            contrastive_loss * contrast_mask
        ).sum() / (contrast_mask.sum() + 1e-5)

        # Consistency loss training: Sample short segments from the audio
        # and extract their embeddings.
        if self.slicer is not None:
            if not self.training:
                generator = torch.Generator().manual_seed(0)
            else:
                generator = None
            # Extract and encode slices detached from context
            latents_slices, seq_len_slices, indices = self.slicer(
                latents, seq_len_y, rng=generator,
            )
            _, embds_slices, seq_len_slices = self.encode(
                None, None,
                latents=latents_slices,
                seq_len_latents=seq_len_slices.to(latents_slices.device),
            )
            seq_len_slices = to_numpy(seq_len_slices, copy=True)
            embds_slices = self.normalize_encoder_output(
                embds_slices, seq_len_slices
            )
            y_slices, seq_len_y_slice = self.transform(
                embds_slices, seq_len_slices, enforce_sorted=False,
            )
            _, fpreds_slices = self.project(y_slices, seq_len_y_slice)

            # Gather reference embeddings from full context embeddings
            indices = indices[..., 0]  # Remove feature dimension
            indices = pt.pad_sequence(indices, batch_first=True)
            while indices.ndim < embds.ndim:
                indices = indices.unsqueeze(-1)
            indices = indices.expand(embds_slices.shape)
            embds_ref = embds.gather(1, indices)

            # Consistency loss on embeddings
            consistency_loss = F.mse_loss(
                embds_ref, embds_slices, reduction='none',
            ).mean(-1)
            consistency_loss = Mean(axis=-1)(consistency_loss, seq_len_slices)
            summary["losses"]["consistency_loss_emb"] = consistency_loss.mean()
            # Consistency loss on frame-level predictions
            fpreds_ref = frame_preds.gather(1, indices[..., 0])
            fpreds_diff = F.l1_loss(fpreds_ref, fpreds_slices, reduction='none')
            fpreds_diff = Mean(axis=-1)(fpreds_diff, seq_len_y_slice)
            summary["losses"]["consistency_loss_scores"] = fpreds_diff.mean()

        with torch.no_grad():
            # Track frame-level volatility
            log_returns = torch.log(
                self.inverse_normalization(frame_preds)
            ).diff()
            log_returns = pt.unpad_sequence(log_returns.T, seq_len_y)
            sigma_returns = torch.stack(list(map(torch.std, log_returns)))
            frame_lens = torch.from_numpy(np.asarray(seq_len_y))\
                .to(sigma_returns.device)
            volatility = sigma_returns * torch.sqrt(
                frame_lens/self.encoder.frame_rate
            )
            summary["scalars"]["volatility"] = volatility.mean().item()
            # Add to buffer to evaluate pseudo-global metrics later
            summary["buffers"]["preds"] = to_numpy(preds, detach=True)
            summary["buffers"]["targets"] = to_numpy(targets)
            summary["histograms"].update({
                "preds_": preds.flatten(),
                "targets_": targets.flatten(),
                "frame_scores_": torch.cat(
                    pt.unpad_sequence(frame_preds.moveaxis(1, 0), seq_len_y)
                ),
            })

        summary = self.finalize_summary(summary)
        return summary

    def modify_summary(self, summary: dict):
        if "preds" in summary["buffers"]:
            preds = np.concatenate(summary["buffers"].pop("preds"))
            targets = np.concatenate(summary["buffers"].pop("targets"))
            corr_mat = np.corrcoef(targets, preds)
            assert corr_mat.shape == (2, 2), corr_mat.shape
            lcc = np.nan_to_num(corr_mat[0, 1], nan=0)
            srcc = np.nan_to_num(
                scipy.stats.spearmanr(targets, preds)[0], nan=0,
            )
            ktau = np.nan_to_num(
                scipy.stats.kendalltau(targets, preds)[0],
                nan=0,
            )
            summary["scalars"].update({
                "LCC": lcc,
                "SRCC": srcc,
                "KTAU": ktau,
            })
        return super().modify_summary(summary)


class SpeechQualityPredictor(pt.Module):
    def __init__(
        self,
        ssl_mos: tp.Optional[SSLMOS] = None,
        storage_dir: tp.Optional[tp.Union[str, Path]] = None,
        *,
        checkpoint_name: str = "ckpt_best_SRCC.pth",
        prepare_example_before: bool = True,
        device: tp.Optional[tp.Union[str, torch.device]] = None,
        return_numpy: bool = False,
        median_filter_size: tp.Optional[int] = None,
        consider_mpi: bool = False,
    ):
        super().__init__()
        if ssl_mos is not None:
            self.model = ssl_mos
        elif storage_dir is not None:
            self.model = SSLMOS.from_storage_dir(
                storage_dir, config_name="config.yaml",
                checkpoint_name=checkpoint_name,
                consider_mpi=consider_mpi,
            )
        else:
            raise ValueError(
                "Either ssl_mos or storage_dir must be provided."
            )
        self.model.slicer = None  # Slicer not needed for inference
        self.prepare_example_before = prepare_example_before
        self.device = device
        self.return_numpy = return_numpy
        self.median_filter_size = median_filter_size
        if median_filter_size is not None and median_filter_size % 2 == 0:
            raise ValueError("median_filter_size must be odd.")

        if self.device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        self.model.to(self.device)

    def median_filter(
        self,
        frame_preds: tp.Union[np.ndarray, Tensor],
        sequence_length: tp.Optional[int] = None,
    ):
        if self.median_filter_size is None:
            return frame_preds
        if self.return_numpy:
            if sequence_length is not None:
                frame_preds, pad = np.array_split(
                    frame_preds, [sequence_length]
                )
            else:
                pad = np.zeros(0)
            frame_preds = scipy.signal.medfilt(
                frame_preds, self.median_filter_size
            )
            frame_preds = np.concatenate([frame_preds, pad])
            return frame_preds

        if sequence_length is not None:
            frame_preds, pad = torch.tensor_split(
                frame_preds, [sequence_length]
            )
        else:
            pad = torch.zeros(0, device=frame_preds.device)
        frame_preds = segment_axis(
            frame_preds, length=self.median_filter_size, shift=1,
            end="conv_pad", pad_mode="constant",
        )
        frame_preds = torch.median(frame_preds, dim=-1).values
        hs = (self.median_filter_size - 1) // 2
        frame_preds = frame_preds[hs:-hs or None]
        frame_preds = torch.cat([frame_preds, pad], dim=0)
        return frame_preds

    def stack(self, lst: tp.List[tp.Union[np.ndarray, Tensor]]):
        if self.return_numpy:
            return np.stack(lst)
        return torch.stack(lst)

    @torch.inference_mode()
    def forward(
        self, wavs: tp.Union[np.ndarray, Tensor], num_samples: TSeqLen = None,
    ):
        wavs = self.model.example_to_device(
            {self.model.input_key: wavs}
        )[self.model.input_key].float()
        if self.prepare_example_before:
            wavs = pt.unpad_sequence(wavs.moveaxis(-1, 0), num_samples)
            examples = [{
                self.model.input_key: to_numpy(wav.moveaxis(0, -1)),
                self.model.input_seq_len_key: n_samples,
                "sampling_rate": SAMPLING_RATE,
            } for wav, n_samples in zip(wavs, num_samples)]
            examples = map(self.model.prepare_example, examples)
            batch = self.model.example_to_device(
                collate_fn(list(examples)), self.device
            )
            wavs = batch[self.model.input_key]

        wavs = wavs.to(self.device)
        _, embds, seq_len_embds = self.model.encode(wavs, num_samples)
        embds = self.model.normalize_encoder_output(embds, seq_len_embds)
        y, seq_len_y = self.model.transform(embds, seq_len_embds)  # Decoder embeddings
        preds, frame_preds = self.model.project(y, seq_len_y)

        preds = self.model.inverse_normalization(preds)
        frame_preds = self.model.inverse_normalization(frame_preds)
        if self.return_numpy:
            preds = to_numpy(preds)
            frame_preds = to_numpy(frame_preds)
        frame_preds = self.stack(
            list(map(self.median_filter, frame_preds, seq_len_embds))
        )
        return preds, frame_preds, seq_len_embds
