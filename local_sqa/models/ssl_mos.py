from collections import defaultdict
from pathlib import Path
import typing as tp

import numpy as np
from paderbox.utils.nested import nested_merge
import padertorch as pt
from padertorch.contrib.je.modules.conv import CNN1d
from padertorch.contrib.je.modules.reduce import Mean
from padertorch.contrib.mk.typing import TSeqLen
from padertorch.data.segment import segment_axis
from padertorch.ops.mappings import ACTIVATION_FN_MAP
from padertorch.utils import to_numpy
import scipy
import torch
from torch import Tensor, nn
import torch.nn.functional as F

SAMPLING_RATE = 16_000


def is_nan(x: tp.Union[Tensor, np.ndarray, float]) -> bool:
    if isinstance(x, Tensor):
        return torch.isnan(x).all().item()
    else:
        return np.isnan(x).all()


class SSLMOS(pt.Model):
    def __init__(
        self,
        encoder: pt.Module,
        criterion,
        proj_in_size: int,
        loss_weights=None,
        decoder: tp.Optional[pt.Module] = None,
        batchformer: tp.Optional[pt.Module] = None,
        out_activation=None,
        input_key: str = "audio",
        input_seq_len_key: str = "num_samples",
        target_key: str = "rating",
        scale: float = 1.,
        shift: float = 0.,
        margin: float = 0.1,
        mae_clip: float = 0.1,
        normalize_ratings: bool = True,
        standardize_audio: bool = True,
        equal_loudness: bool = True,
        l2_normalization: bool = True,
        normalize_decoder: bool = False,
        zero_init: bool = False,
        slicer: tp.Optional[pt.Module] = None,
        scl: tp.Optional[pt.Model] = None,
        forget_gate_bias: tp.Optional[float] = None,
        bias: bool = True,
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
        self.batchformer = batchformer
        if proj_in_size > 1:
            self.out_proj = nn.Linear(proj_in_size, 1, bias=bias)

        self.input_key = input_key
        self.input_seq_len_key = input_seq_len_key
        self.target_key = target_key
        self.scale = scale
        self.shift = shift
        self.margin = margin
        self.mae_clip = mae_clip
        self.normalize_ratings = normalize_ratings
        self.standardize_audio = standardize_audio
        self.equal_loudness = equal_loudness
        self.l2_normalization = l2_normalization
        self.normalize_decoder = normalize_decoder
        self.slicer = slicer
        self.scl = scl

        self.reset_parameters()
        if zero_init:
            self.zero_init_()

    @classmethod
    def finalize_dogmatic_config(cls, config):
        config["criterion"] = {
            'factory': nn.L1Loss,
            'reduction': 'none',
        }

    def _normalize_ratings(self, ratings: np.ndarray):
        ratings = (ratings - 1) / 2 - 1  # [-1, 1]
        if self.batchformer is None:
            return ratings
        # Batch-relative normalization
        triplets = segment_axis(
            ratings, length=3, shift=3, axis=0, end="pad", pad_mode="edge"
        )
        max_rating = triplets.max(axis=1, keepdims=True)
        min_rating = triplets.min(axis=1, keepdims=True)
        triplets = (
            (triplets - min_rating) / (max_rating - min_rating + 1e-6) * 2 - 1
        )  # [-1, 1]
        triplets = triplets.reshape(-1)[:ratings.shape[0]]
        return np.stack((ratings, triplets), axis=0)

    def _normalize(self, x: Tensor, seq_len_x: TSeqLen = None):
        m = Mean(axis=1, keepdims=True)(x.detach(), seq_len_x)
        norm = torch.linalg.norm(m, ord=2, dim=-1, keepdim=True)
        x = x / (norm + 1e-6)
        return x

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
        # May raise a UserWarning: The given NumPy array is not writable ...
        # Caused from self.input_key. Can be ignored since we don't modify the
        # input audio.
        example = super().example_to_device(example, device, memo)
        audio = example[self.input_key]
        if isinstance(audio, list):
            audio = pt.pad_sequence(audio, batch_first=True)
            example[self.input_key] = audio
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

    def extract_ratings(self, inputs: dict):
        ratings = inputs
        for k in self.target_key.split("."):
            ratings = ratings[k]
        ratings = np.asarray(ratings)
        if self.normalize_ratings:
            ratings = self._normalize_ratings(ratings)
        # ratings = torch.tensor(ratings)
        return ratings

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

    def decode(
        self, decoder, x, seq_len_x, enforce_sorted=True
    ):
        if decoder is not None:
            chunks = 1
            if x.ndim == 4:
                xs = x.unbind(dim=1)
                chunks = len(xs)
                x = torch.cat(xs, dim=0)
                if seq_len_x is not None:
                    seq_len_x = np.concatenate(
                        np.split(seq_len_x, chunks, axis=1), axis=0
                    ).squeeze(1)
            if isinstance(decoder, nn.LSTM):
                if seq_len_x is not None:
                    x = pt.pack_padded_sequence(
                        x, seq_len_x, batch_first=True,
                        enforce_sorted=enforce_sorted,
                    )
                x, _ = decoder(x)
                if seq_len_x is not None:
                    x, _ = pt.pad_packed_sequence(x, batch_first=True)
            elif isinstance(decoder, CNN1d):
                x, seq_len_x = decoder(x.moveaxis(2, 1), seq_len_x)
                x = x.moveaxis(1, 2)
            else:
                x, _ = decoder(x, seq_len_x)
            if chunks > 1:
                x = torch.stack(torch.chunk(x, chunks, dim=0), dim=1)
                seq_len_x = np.stack(
                    np.split(seq_len_x, chunks, axis=0), axis=1
                )
        return x, seq_len_x

    def normalize_encoder_output(
        self,
        x: Tensor,
        seq_len_x: TSeqLen = None,
    ) -> Tensor:
        if self.l2_normalization:
            # Keeps losses low at initialization
            x = self._normalize(x, seq_len_x)
        return x

    def transform(
        self, x: Tensor,
        seq_len_x: TSeqLen = None,
        enforce_sorted=True
    ):
        x_, seq_len_x = self.decode(
            self.decoder, x, seq_len_x, enforce_sorted=enforce_sorted
        )
        return x_, seq_len_x

    def reduce(
        self, x: Tensor, seq_len_x: TSeqLen,
    ):
        if x.ndim == 3:
            if seq_len_x is not None and not isinstance(seq_len_x, Tensor):
                seq_len_x = torch.tensor(seq_len_x).long().to(x.device)
            return torch.vmap(
                Mean(axis=1), in_dims=1, out_dims=1
            )(x, seq_len_x)
        return Mean(axis=1)(x, seq_len_x)

    def project(self, x: Tensor, seq_len_x: TSeqLen):
        if hasattr(self, 'out_proj'):
            x = self.out_proj(x)
        preds = self.out_activation(x).squeeze(-1)
        preds = preds*self.scale+self.shift
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
        if self.input_seq_len_key is not None:
            sequence_lengths = inputs[self.input_seq_len_key]
        else:
            sequence_lengths = [wavs.shape[-1]] * wavs.shape[0]

        ratings = self.extract_ratings(inputs)
        latents, embds, seq_len_embds = self.encode(
            wavs, sequence_lengths
        )
        embds = self.normalize_encoder_output(embds, seq_len_embds)
        # loss_mask = torch.ones(ratings.shape).float().to(embds.device)
        y, seq_len_y = self.transform(embds, seq_len_embds)  # Decoder embeddings
        preds, frame_preds = self.project(y, seq_len_y)
        return (
            frame_preds, seq_len_embds, preds, (latents, embds), ratings,
            # loss_mask
        )

    def review(self, inputs: dict, outputs: dict):
        summary = {'losses': {}, 'histograms': {}, 'buffers': {}, 'scalars': {}}
        summary["scalars"]["scale"] = self.scale
        summary["scalars"]["shift"] = self.shift

        # Prepare targets
        frame_preds, seq_len_y, preds, (latents, embds), targets = outputs
        targets = torch.from_numpy(targets).to(preds.device)
        if preds.ndim == 0:
            preds = preds.unsqueeze(0)
        while targets.ndim < preds.ndim:
            targets = targets.unsqueeze(-1)
        if targets.ndim != preds.ndim:
            raise ValueError(
                f"Expected targets to have shape {preds.shape}, "
                f"got {targets.shape}"
            )
        targets = targets.expand_as(preds)
        assert preds.shape == targets.shape, (preds.shape, targets.shape)

        # Utterance-level regression loss + contrastive loss
        regr = self.criterion(preds, targets)
        regr_mask = ((preds-targets).abs() > self.mae_clip)
        summary["losses"]["regression"] = (
            (regr*regr_mask).sum()/(regr_mask.sum()+1e-6)
        )
        if preds.ndim > 1:
            preds = preds[:, 0]
            targets = targets[:, 0]
            frame_preds = frame_preds[:, 0]
            seq_len_y = seq_len_y[:, 0]
            embds = embds[:, 0]
            summary["scalars"]["triplet_regression"] = (
                (regr[:, 1]*regr_mask[:, 1]).sum()/(regr_mask[:, 1].sum()+1e-6)
            ).item()
        # Contrastive loss
        diff = targets[:, None] - targets[None]
        diff_preds = preds[:, None] - preds[None]
        contrastive_loss = torch.relu((diff-diff_preds).abs()-self.margin)
        if contrastive_loss.ndim > 2:
            contrastive_loss = contrastive_loss.moveaxis(2 ,0)
            diff = diff.moveaxis(2, 0)
        contrastive_loss = torch.triu(contrastive_loss)
        contrast_mask = torch.triu(torch.ones_like(diff).to(preds.device))
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
                audio_start_samples=self.encoder.sample_index_to_frame_index(
                    inputs.get("audio_start_samples", None)
                ),
                audio_stop_samples=self.encoder.sample_index_to_frame_index(
                    inputs.get("audio_stop_samples", None)
                ),
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
            # Decode slice embeddings
            y_slices, seq_len_slices = self.transform(
                embds_slices, seq_len_slices,
                enforce_sorted=False,
            )
            # Gather reference embeddings from full context embeddings
            embds_ref, embds_slices, seq_len_slices_ = self.slicer.reshape(
                embds, embds_slices, indices, seq_len_slices
            )
            # Consistency loss on embeddings
            consistency_loss = F.mse_loss(
                embds_ref, embds_slices, reduction='none',
            ).mean(-1)
            consistency_loss = Mean(axis=-1)(consistency_loss, seq_len_slices_)
            summary["losses"]["consistency_loss_emb"] = consistency_loss.mean()
            # Consistency loss on frame-level predictions
            _, fpreds_slices = self.project(y_slices, seq_len_slices)
            if self.slicer.detach:
                fpreds_slices = fpreds_slices.detach()
            fpreds_ref, fpreds_slices, _ = self.slicer.reshape(
                frame_preds, fpreds_slices, indices, seq_len_slices
            )
            fpreds_diff = F.l1_loss(
                fpreds_ref, fpreds_slices, reduction='none'
            )
            fpreds_diff = Mean(axis=-1)(fpreds_diff, seq_len_slices_)
            summary["losses"]["consistency_loss_scores"] = fpreds_diff.mean()

        if not all(
            is_nan(frame_mos) for frame_mos in inputs.get("frame_mos", [])
        ):
            # Optional frame-level supervision
            frame_targets_mask = [
                not is_nan(ft) for ft in inputs["frame_mos"]
            ]
            seq_len_y_ = seq_len_y[frame_targets_mask]
            frame_preds_ = frame_preds[frame_targets_mask, :seq_len_y_.max()]
            frame_targets = [
                ft for ft, m in zip(inputs["frame_mos"], frame_targets_mask)
                if m
            ]
            frame_targets = self._normalize_ratings(pt.pad_sequence(
                frame_targets, batch_first=True
            ))
            frame_regression = self.criterion(frame_preds_, frame_targets)
            frame_regression = Mean(axis=-1)(frame_regression, seq_len_y_)
            summary["losses"]["frame_regression"] = frame_regression.mean()

        if self.scl is not None:
            # Supervised contrastive learning on embeddings
            if self.normalize_decoder:
                embds = embds / (
                    torch.norm(embds, p=2, dim=-1, keepdim=True) + 1e-6
                )
            scl_inputs = {
                **inputs,
                self.scl.input_key: embds,
                self.scl.input_seq_len_key: seq_len_y,
            }
            scl_summary = self.scl.review(scl_inputs, self.scl(scl_inputs))
            summary = nested_merge(summary, scl_summary, allow_update=False)

        with torch.no_grad():
            if preds.ndim > 1:
                preds = preds[..., 0]
                targets = targets[..., 0]
                frame_preds = frame_preds[..., 0]
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
        if self.scl is not None:
            summary = self.scl.modify_summary(summary)
        return super().modify_summary(summary)


class SpeechQualityPredictor(pt.Module):
    @classmethod
    def from_state_dict(cls, state_dict: dict, config: dict, **kwargs):
        model = SSLMOS.from_config(config)
        model.load_state_dict(state_dict)
        return cls(ssl_mos=model, **kwargs)

    def __init__(
        self,
        ssl_mos: tp.Optional[SSLMOS] = None,
        storage_dir: tp.Optional[tp.Union[str, Path]] = None,
        *,
        checkpoint_name: str = "ckpt_best_SRCC.pth",
        prepare_example_before: bool = False,
        device: tp.Optional[tp.Union[str, torch.device]] = None,
        return_numpy: bool = False,
        median_filter_size: tp.Optional[int] = None,
        consider_mpi: bool = False,
        weights_only: bool = True,
    ):
        super().__init__()
        if ssl_mos is not None:
            self.model = ssl_mos
        elif storage_dir is not None:
            self.model = SSLMOS.from_storage_dir(
                storage_dir, config_name="config.yaml",
                checkpoint_name=checkpoint_name,
                consider_mpi=consider_mpi,
                weights_only=weights_only,
            )
        else:
            raise ValueError(
                "Either ssl_mos or storage_dir must be provided."
            )
        self.model.slicer = None  # Slicer not needed for inference
        self.prepare_example_before = prepare_example_before
        if self.prepare_example_before:
            raise NotImplementedError(
                "prepare_example_before=True is not implemented."
            )
        self.device = device
        self.return_numpy = return_numpy
        self.median_filter_size = median_filter_size
        if median_filter_size is not None and median_filter_size % 2 == 0:
            raise ValueError("median_filter_size must be odd.")

        if self.device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        self.model.to(self.device).eval()

    def median_filter(
        self,
        frame_preds: tp.Union[np.ndarray, Tensor],
        sequence_length: tp.Optional[int] = None,
    ):
        if self.median_filter_size is None or self.median_filter_size <= 1:
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
        self,
        wavs: tp.Union[np.ndarray, Tensor],
        num_samples: TSeqLen = None,
        return_embeddings: bool = False,
        embedding_type: str = "decoder",
    ):
        wavs = self.model.example_to_device(
            {self.model.input_key: wavs}
        )[self.model.input_key].float()

        wavs = wavs.to(self.device)
        _, embds, seq_len_embds = self.model.encode(wavs, num_samples)
        embds = self.model.normalize_encoder_output(
            embds, seq_len_embds
        )

        if return_embeddings:
            if embedding_type in ("decoder", "projection"):
                if self.model.scl is not None:
                    _, embds, *_ = self.model.scl.encode(
                        embds, seq_len_embds
                    )
                    if embedding_type == "projection":
                        embds = self.model.scl.proj(embds)
                else:
                    embds, _ = self.model.transform(embds, seq_len_embds)
            elif embedding_type != "encoder":
                raise ValueError(
                    f"Invalid embedding_type {embedding_type}, "
                    f"expected 'encoder', 'decoder' or 'projection'."
                )
            if self.return_numpy:
                embds = to_numpy(embds)
            return embds, seq_len_embds

        y, seq_len_y = self.model.transform(embds, seq_len_embds)  # Decoder embeddings
        preds, frame_preds = self.model.project(y, seq_len_y)

        if preds.ndim > 1:
            preds = preds[..., 0]
            frame_preds = frame_preds[..., 0]
        preds = self.model.inverse_normalization(preds)
        frame_preds = self.model.inverse_normalization(frame_preds)
        if self.return_numpy:
            preds = to_numpy(preds)
            frame_preds = to_numpy(frame_preds)
        if seq_len_embds is None:
            seq_len_embds = [frame_preds.shape[0]] * frame_preds.shape[1]
        frame_preds = self.stack(
            list(map(self.median_filter, frame_preds, seq_len_embds))
        )
        return preds, frame_preds, seq_len_embds


class TeacherStudent(SSLMOS):
    def __init__(
        self,
        *args,
        teacher: SpeechQualityPredictor,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.teacher = teacher

    def forward(self, inputs: dict):
        input_key = self.input_key
        input_seq_len_key = self.input_seq_len_key
        wavs = inputs[input_key]
        num_samples = inputs.get(input_seq_len_key, None)
        with torch.inference_mode():
            _, teacher_frame_preds, seq_len_embds = self.teacher(
                wavs, num_samples, return_embeddings=False,
            )
        frame_mos = inputs.get("frame_mos", [])
        frame_mos = [
            teacher_frame_preds[i, :seq_len_embds[i]] if not is_nan(fm) else fm
            for i, fm in enumerate(frame_mos)
        ]
        inputs["frame_mos"] = frame_mos
        return inputs, super().forward(inputs)

    def review(self, inputs: dict, outputs: dict):
        inputs, outputs = outputs
        return super().review(inputs, outputs)
