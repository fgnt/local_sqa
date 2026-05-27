import functools
import typing as tp
import warnings

from einops import rearrange
import numpy as np
from paderbox.utils.nested import flatten, deflatten
import padertorch as pt
from padertorch.contrib.je.modules.conv import CNN1d, CNNTranspose1d
from padertorch.contrib.je.modules.reduce import Mean
from padertorch.contrib.mk.typing import TSeqLen
from padertorch.contrib.tcl.speaker_embeddings.eer_metrics import get_eer
from padertorch.ops.mappings import _CallableDispatcher
from padertorch.ops.sequence.mask import compute_mask
from padertorch.data.utils import pad_tensor
from scipy.spatial.distance import cosine
import torch
import torch.nn as nn
import torch.nn.functional as F


REDUCE_MAP = _CallableDispatcher(
    mean=Mean,
)


def add_prefix(name: str, mapping: dict, allow_update: bool = True):
    """
    Prefix keys at the lowest hierarchy level of mapping

    >>> mapping = {'audio_path': {'observation': 'observation.wav'}, 'gender': 'f'}
    >>> add_prefix('source', mapping, allow_update=False)
    {'audio_path': {'source_observation': 'observation.wav'},
     'gender': {'source': 'f'}}
    >>> add_prefix('source', mapping, allow_update=True)
    {'audio_path': {'source_observation': 'observation.wav'}, 'gender': 'f'}

    Args:
        name: The prefix string
        mapping: Nested mapping
        allow_update: If True, only add prefix to keys at the lowest hierarchy
            level. If False, create a new hierarchy level for non-nested keys
            to avoid updates by `nested_merge(..., allow_update=True)`.

    Returns: dict
        Mapping with prefixed keys
    """
    if name.endswith('_'):
        name = name[:-1]
    if len(name) == 0:
        return mapping

    def _process_key(_key):
        *path, stem = _key.rsplit('.')
        if len(path) == 0 and not allow_update:
            # Create new hierarchy level
            stem = '.'.join((stem, name))
        elif len(path) > 0:
            stem = '_'.join((name, stem))
        return '.'.join((*path, stem))

    return deflatten(
        dict(list(map(
            lambda item: (k := item[0], v := item[1], (_process_key(k), v))[-1],
            flatten(mapping, sep='.').items(),
        ))), sep='.',
    )


def cosine_similarity(tensor, other, eps=1e-12):
    """

    Args:
        tensor: Shape (..., b, d)
        other: Shape (..., d, b)
        eps:

    Returns:

    """
    prod = torch.matmul(tensor, other)
    norm = (
        torch.linalg.norm(tensor, ord=None, dim=-1, keepdim=True)
        * torch.linalg.norm(other, ord=None, dim=-2, keepdim=True)
    )
    return prod / torch.maximum(norm, torch.tensor(eps).to(norm.device))


SIMILARITIES = _CallableDispatcher(
    dot_product=torch.matmul,
    cosine=cosine_similarity,
)


class SCL(pt.Model):
    """Supervised Contrastive Learning.

    Official implementation: https://github.com/HobbitLong/SupContrast

    Args:
        encoder: Embedding extractor network.
        label_key (str): Key which stores affilation of input to a class. Inputs
            from the same class will be encoded close to each other and further
            apart from other classes.
        feature_extractor (callable, optional): Feature extractor on input 
            signal. If multi_view=True and second_view_extractor=None, should
            support random augmentations to create different views of the same
            input.
        multi_view (bool): If True, extract a second feature view from the input
            signal.
        second_view_extractor (callable, optional): If not None, extract
            a second feature view from the input signal.
        l2_normalization (bool): If True, perform L2-normalization of encoder
            embeddings.
        temperature (float): Value to scale logits by before applying softmax.
        similarity: Similarity function to use for computing logits. Supported
            are "dot_product" and "cosine" (L2-normalized dot product).
        input_key:
        input_seq_len_key:
        prefix:
    """
    def __init__(
        self, encoder, *,
        label_key: str,
        embedding_size: int,
        proj_size: tp.Optional[int] = None,
        feature_extractor: tp.Optional[pt.Module] = None,
        multi_view: bool = False,
        second_view_extractor: tp.Optional[pt.Module] = None,
        l2_normalization: bool = False,
        temperature: float = 1.,
        similarity: str = 'dot_product',
        input_key: str = 'x',
        input_seq_len_key: tp.Optional[str] = "seq_len_x",
        target_seq_len_key: tp.Optional[str] = None,
        prefix: tp.Optional[str] = None,
        sequence_axis: int = 1,
        negative_to_target_dist: int = 0,
        positive_to_target_dist: int = 0,
        normalize_encoder: bool = False,
        ignore_index: int = -100,
        reduce: tp.Optional[pt.Module] = None,
        negatives_mode: str = "time",
        input_shape: str = "b t d",
        final_batch_norm: bool = False,
        add_ignore_to_negatives: bool = True,
        distance_metric: str = "cosine",
        classification: bool = False,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.encoder = encoder
        self.embedding_size = embedding_size

        if proj_size is not None:
            self.projection_head = nn.Linear(
                embedding_size, proj_size, bias=False
            )
        else:
            self.projection_head = None

        self.proj_size = proj_size
        self.label_key = label_key
        self.multi_view = multi_view
        self.second_view_extractor = second_view_extractor
        self.l2_normalization = l2_normalization
        self.temperature = temperature
        self.input_key = input_key
        self.input_seq_len_key = input_seq_len_key
        self.target_seq_len_key = target_seq_len_key
        self.prefix = '' if prefix is None else prefix + '_'
        self.sequence_axis = sequence_axis
        self.negative_to_target_dist = negative_to_target_dist
        self.positive_to_target_dist = positive_to_target_dist
        self.normalize_encoder = normalize_encoder
        self.ignore_index = ignore_index
        self.negatives_mode = negatives_mode
        self.input_shape = input_shape
        self.add_ignore_to_negatives = add_ignore_to_negatives
        self.distance_metric = distance_metric
        self.classification = classification
        if final_batch_norm:
            self.bn = nn.BatchNorm1d(embedding_size, affine=False)
        else:
            self.bn = None
        if reduce is not None:
            self.reduce = REDUCE_MAP[reduce](sequence_axis=sequence_axis)
        else:
            self.reduce = None

        self._similarity = similarity
        self.similarity = SIMILARITIES[similarity]

    def _rearrange(
        self,
        tensor: torch.Tensor,
        input_shape: str,
        sequence_last: bool = False,
    ) -> tp.Tuple[torch.Tensor, str]:
        if sequence_last and not input_shape.endswith('t'):
            target_shape = input_shape.split()
            target_shape.remove('t')
            target_shape.append('t')
        elif not sequence_last and input_shape.endswith('t'):
            target_shape = input_shape.split()
            target_shape.remove('t')
            target_shape.insert(-1, 't')
        else:
            return tensor, input_shape
        target_shape = " ".join(target_shape)
        pattern = input_shape + ' -> ' + target_shape
        return rearrange(tensor, pattern), target_shape

    def labels2tensor(self, labels, max_seq_len, device):
        if isinstance(labels, list):
            if isinstance(labels[0], list):
                # Multiple labels per example (sequence labeling)
                if any(
                    len(li) > 0 and isinstance(li[0], list) for li in labels
                ):
                    # Multiple classes per time step
                    unq = {lab for li in labels for lti in li for lab in lti}
                    if any(isinstance(lab, str) for lab in unq):
                        # label2id = dict(zip(unq, range(len(unq))))
                        label2id = {
                            lab: i if lab != self.ignore_index else
                            self.ignore_index for i, lab in enumerate(unq)
                        }
                    else:
                        label2id = {lab: lab for lab in unq}
                    labels = [
                        [
                            torch.tensor(
                                list(map(label2id.get, lti)), device=device
                            )
                            for lti in li
                        ]
                        for li in labels
                    ]
                    labels = [
                        pt.pad_sequence(
                            li, batch_first=True,
                            padding_value=self.ignore_index
                        ) if len(li) > 0 else torch.tensor([], device=device)
                        for li in labels
                    ]
                    # Unsqueeze empty label lists
                    labels = [
                        li if li.ndim > 1 else torch.empty([0,0], device=device)
                        for li in labels
                    ]
                    max_num_classes = max(li.shape[1] for li in labels)
                    labels = [
                        F.pad(
                            li, (0, max_num_classes - li.shape[1]),
                            value=self.ignore_index
                        ) for li in labels
                    ]
                    labels = pt.pad_sequence(
                        labels, batch_first=True,
                        padding_value=self.ignore_index
                    )
                else:
                    # One class per time step
                    unq = {lab for li in labels for lab in li}
                    if any(isinstance(lab, str) for lab in unq):
                        # label2id = dict(zip(unq, range(len(unq))))
                        label2id = {
                            lab: i if lab != self.ignore_index else
                            self.ignore_index for i, lab in enumerate(unq)
                        }
                    else:
                        label2id = {lab: lab for lab in unq}
                    labels = [
                        torch.tensor(list(map(label2id.get, li)), device=device)
                        for li in labels
                    ]
                    labels = pt.pad_sequence(
                        labels, batch_first=True,
                        padding_value=self.ignore_index
                    )
            elif any(isinstance(lab, str) for lab in labels):
                # One label per example
                try:
                    labels = [int(label) for label in labels]
                except ValueError:
                    unq = np.unique(labels)
                    # label2id = dict(zip(unq, range(len(unq))))
                    label2id = {
                        lab: i if lab != self.ignore_index else
                        self.ignore_index for i, lab in enumerate(unq)
                    }
                    labels = list(map(label2id.get, labels))
                labels = torch.tensor(labels, device=device)
            else:
                # Expect integer or float labels
                labels = torch.tensor(labels, device=device)
        elif isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels).to(device)
        elif not isinstance(labels, torch.Tensor):
            raise TypeError(type(labels))
        if labels.shape[1] < max_seq_len:
            if labels.ndim == 3:
                labels = F.pad(
                    labels, (0, 0, 0, max_seq_len - labels.shape[1]),
                    value=self.ignore_index
                )
            else:
                labels = F.pad(
                    labels, (0, max_seq_len - labels.shape[1]),
                    value=self.ignore_index
                )
        return labels

    def encode(self, x, seq_len_x, target_shape=None, inputs=None):
        if self.multi_view:
            if self.second_view_extractor is not None:
                x_targets, seq_len_targets = self.second_view_extractor(x, seq_len=seq_len_x)
            elif self.feature_extractor is not None:
                x_targets, seq_len_targets = self.feature_extractor(x, seq_len=seq_len_x)
            else:
                raise ValueError(
                    'multi_view training requires two different views, i.e. '
                    'two separate feature_extractors or one feature_extractor '
                    'with data augmentation.'
                )
        else:
            x_targets = None

        if self.feature_extractor is not None:
            x, seq_len_x = self.feature_extractor(x, seq_len=seq_len_x)
        if x_targets is not None:
            x = torch.cat((x, x_targets), dim=0)
            assert (np.array(seq_len_x) == seq_len_targets).all()
            seq_len_x = np.concatenate((seq_len_x, seq_len_targets))
        if isinstance(self.encoder, CNNTranspose1d):
            if target_shape is not None:
                target_shape = list(target_shape)
                target_shape[1] = self.encoder.out_channels[-1]
            kwargs = dict(target_shape=target_shape)
        else:
            kwargs = {}
        if self.encoder is None:
            z, seq_len_z = x, seq_len_x
            z_shape = self.input_shape
        else:
            if self.normalize_encoder:
                for _, module in self.encoder.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        module.weight.data = F.normalize(
                            module.weight, p=2, dim=1
                        )
                    elif isinstance(module, torch.nn.Conv1d):
                        module.weight.data = F.normalize(
                            module.weight.view(module.out_channels, -1),
                            p=2, dim=1
                        ).view_as(module.weight)
            x, z_shape = self._rearrange(
                x, self.input_shape, sequence_last=True
            )
            if (
                isinstance(self.encoder, (CNN1d, CNNTranspose1d))
                and x.dim() != 3
            ):
                assert x.dim() == 4, x.dim()
                x = rearrange(x, 'b d f t -> b (d f) t')
            z, seq_len_z = self.encoder(x, seq_len_x, **kwargs)
        if z.dim() == 4:
            z = rearrange(z, 'b d f t -> b (d f) t')
            z_shape = 'b d t'

        if self.reduce is not None:
            z, seq_len_z, loss_weights = self.reduce(
                z, seq_len_z, inputs=inputs, return_weights=True,
            )
        else:
            loss_weights = None

        if self.bn is not None:
            z, z_shape = self._rearrange(
                z, z_shape, sequence_last=True
            )
            z = self.bn(z)

        z, z_shape = self._rearrange(
            z, z_shape, sequence_last=False
        )

        return x, z, seq_len_z, z_shape, loss_weights

    def proj(self, z):
        if self.l2_normalization:
            z = torch.div(
                z, torch.norm(z, p=2, dim=-1, keepdim=True) + 1e-5
            )
        if self.projection_head is not None:
            if self.l2_normalization:
                self.projection_head.weight.data = torch.nn.functional\
                    .normalize(self.projection_head.weight, p=2, dim=1)
            z = self.projection_head(z)
        return z

    def contrast(
        self,
        z_proj: torch.Tensor,
        labels: tp.Union[tp.List, np.ndarray, torch.Tensor],
        sequence_lengths: TSeqLen = None,
    ):
        """Compute supervised contrastive loss.

        Implementation based on
        https://github.com/HobbitLong/SupContrast/blob/master/losses.py.

        Args:
            z_proj (torch.Tensor): Embeddings at the output of the encoder.
                Shape (b, n_views, ..., d) or (b, n_views, l, ..., d).
            labels (torch.Tensor): Class labels of shape (b,) or (b, l).
        """
        device = z_proj.device

        if labels.numel() == 0:
            return None, None, None, None

        contrast_count = z_proj.shape[1]
        contrast_feature = torch.cat(torch.unbind(z_proj, dim=1), dim=0)

        if labels.ndim == 1:
            mask = (labels[None] == labels[:, None]).float()  # Target mask for loss computation, shape (b, b)
            anchor_feature = rearrange(contrast_feature, 'b ... d -> ... b d')
            contrast_feature = rearrange(contrast_feature, 'b ... d -> ... d b')
            reshape = '... b k -> b k ...'
            # Mask out self-contrast cases
            mask = mask.repeat(
                contrast_count, contrast_count,
                # *[1]*(mask.ndim-2)
            )
            logits_mask = torch.ones_like(mask)  # Samples to contrast with
            # Only keep negative samples
            # logits_mask[torch.eye(batch_size*contrast_count).bool()] = 0
            logits_mask = logits_mask - mask
            # No loss on ignored labels
            ignore_mask = (labels[:, None] != self.ignore_index).float()
            padding_mask = torch.ones_like(logits_mask)
        else:
            b, l, *_ = labels.shape
            contrast_feature = contrast_feature[:, :l]
            anchor_feature = rearrange(
                contrast_feature, "b l ... d -> b ... l d"
            )
            # Ignore padding labels
            labels = labels * compute_mask(
                labels, sequence_lengths, sequence_axis=1
            ).long()
            labels += (1-compute_mask(
                labels, sequence_lengths, sequence_axis=1
            ).long()) * self.ignore_index
            stride = (
                self.negative_to_target_dist
                if self.negative_to_target_dist > 0
                else self.positive_to_target_dist
                if self.positive_to_target_dist > 0 else 1
            )
            pos = (
                torch.arange(b)[:, None] * stride * l + torch.arange(l)
            ).to(device)  # (b, l)
            if self.negatives_mode == "time":
                contrast_feature = rearrange(
                    contrast_feature, "b l ... d -> b ... d l"
                )
                if labels.ndim == 2:
                    mask = (labels[:, None] == labels[:, :, None]).float()  # shape (b, l, l, ...)
                    ignore_mask = (
                        labels[:, :, None] != self.ignore_index
                    ).float()
                else:
                    raise NotImplementedError(labels.ndim)
                # Mask out self-contrast cases
                self_contrast = torch.eye(mask.shape[1]).to(device).unsqueeze(0)
                while self_contrast.ndim < mask.ndim:
                    self_contrast = self_contrast.unsqueeze(-1)
                self_contrast = self_contrast.expand_as(mask)
                padding_mask = compute_mask(
                    mask, sequence_lengths, sequence_axis=1
                )
                pos_diff = torch.abs(pos[:, None] - pos[:, :, None])
            elif self.negatives_mode == "all":
                contrast_feature = rearrange(
                    contrast_feature, "b l ... d -> ... d (b l)"
                )
                contrast_labels = rearrange(labels, "b l ... -> (b l) ...")
                # Target mask for loss computation
                if labels.ndim == 2:
                    mask = (labels[:, None] == contrast_labels[None, :, None])\
                        .float()  # shape (b, b*l, l, ...)
                    # No loss on ignored labels
                    ignore_mask = (
                        contrast_labels[None, :, None] != self.ignore_index
                    ).float() * (labels[:, None] != self.ignore_index).float()
                else:
                    # Multiple classes per time step
                    mask = torch.stack([
                        torch.isin(contrast_labels, li[li!=self.ignore_index])
                        for li in contrast_labels
                    ]).any(dim=-1).float()
                    mask = rearrange(mask, "(b l) ... -> b ... l", b=b, l=l)
                    ignore_mask = torch.all(
                        contrast_labels == self.ignore_index, dim=-1
                    ).logical_not().float()[None, :, None]
                    ignore_mask = ignore_mask * torch.all(
                        labels == self.ignore_index, dim=-1
                    ).logical_not().float()[:, None, :]
                # Mask out self-contrast cases
                self_contrast = (
                    pos[:, None]
                    == rearrange(pos, "b l -> (b l)")[None, :, None]
                ).float()
                padding_mask = rearrange(
                    compute_mask(
                        torch.ones(b, l).to(device),
                        sequence_lengths, sequence_axis=1
                    ), "b l -> (b l)"
                ).unsqueeze(0).unsqueeze(-1)
                pos_diff = torch.abs(
                    pos[:, None] - rearrange(pos, "b l -> (b l)")[None, :, None]
                )
            else:
                raise NotImplementedError(self.negatives_mode)
            assert mask.min() >= 0, mask.min()

            if self.negative_to_target_dist > 0:
                logits_mask = (pos_diff > self.negative_to_target_dist).float()
            else:
                # Only remove self-contrast cases
                logits_mask = 1 - self_contrast

            reshape = 'b ... l k -> b k l ...'
            logits_mask = logits_mask * padding_mask.float()
            assert logits_mask.min() >= 0, logits_mask.min()
            if not self.add_ignore_to_negatives:
                logits_mask = logits_mask * ignore_mask

            if self.positive_to_target_dist > 0:
                boundary_mask = pos_diff <= self.positive_to_target_dist
                mask = mask - boundary_mask.float() * mask
            else:
                # Only remove self-contrast cases
                mask = mask - self_contrast * ignore_mask
            assert mask.min() >= 0, mask.min()

        # Compute logits
        anchor_dot_contrast = rearrange(
            torch.div(
                self.similarity(
                    anchor_feature,
                    contrast_feature
                ),
                self.temperature
            ), reshape
        )
        # Increase numerical stability
        # Take max over columns
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Compute log. prob.
        for _ in range(logits.ndim - logits_mask.ndim):
            logits_mask = logits_mask.unsqueeze(-1)
            mask = mask.unsqueeze(-1)
        denum = torch.logsumexp(
            # Add small eps to prevent logsumexp(-inf) when no positive
            # pairs are present
            logits + torch.log(logits_mask + 1e-5), dim=1, keepdim=True
        )
        log_prob = logits - denum

        # Compute mean of log-likelihood over positives
        mask = mask * ignore_mask
        mean_log_prob_pos = (
            (mask * log_prob).sum(1) / (mask.sum(1) + 1e-5)
        )

        # Loss: Minimize negative log-likelihood of positive pairs
        loss = -mean_log_prob_pos
        if torch.any(loss.isnan()):
            breakpoint()

        for _ in range(mask.ndim - 2):
            mask = mask.squeeze(-1)
        loss_mask = (labels != self.ignore_index)
        if loss_mask.ndim == 3:
            loss_mask = loss_mask.any(-1)
        return loss, logits + torch.log(logits_mask), (mask, logits_mask), loss_mask.float()

    def forward(self, inputs):
        x = inputs[self.input_key]
        if self.input_seq_len_key is not None:
            seq_len = inputs[self.input_seq_len_key]
        else:
            seq_len = None
        labels = inputs[self.label_key]
        batch_size = x.shape[0]

        x, z, seq_len_z, z_shape, loss_weights = self.encode(
            x, seq_len_x=seq_len,
            target_shape=inputs.get(f'{self.prefix}target_shape', None),
            inputs=inputs,
        )
        z_proj = self.proj(z)
        if self.multi_view:
            x = torch.stack(torch.split(x, batch_size), dim=1)
            z_proj = torch.stack(torch.split(z_proj, batch_size), dim=1)
            seq_len_z = seq_len_z[:batch_size]
        else:
            z_proj = z_proj.unsqueeze(1)

        labels = self.labels2tensor(labels, max(seq_len_z), z_proj.device)
        loss, _, con_masks, loss_mask = self.contrast(z_proj, labels, seq_len_z)

        return (z, z_proj, labels), loss, loss_mask, loss_weights, con_masks, seq_len_z

    def review(self, inputs, outputs):
        (z, z_proj, labels), loss, loss_mask, loss_weights, con_masks, seq_len_z = outputs
        if loss is None:
            return {}
        if loss_weights is not None:
            loss = loss * loss_weights
        ce = (loss * loss_mask).sum() / (loss_mask.sum() + 1e-5)

        with torch.no_grad():
            positives = con_masks[0].sum(1)  # (b, ...)
            negatives = con_masks[1].sum(1)
            positives = (positives * loss_mask).sum() / (loss_mask.sum() + 1e-5)
            negatives = Mean(axis=1)(negatives, seq_len_z).mean()

        summary = {
            "losses": {"ce": ce},
            "scalars": {
                "positives": positives.item(),
                "negatives": negatives.item(),
            },
            "buffers": {}
        }
        if self.classification:
            if self.projection_head is None:
                raise ValueError("classification requires proj_size to be set.")
            assert labels.max() < self.proj_size, (labels.max(), self.proj_size)
            if z_proj.ndim == 4:
                z_proj = z_proj.squeeze(1)
            if labels.ndim == 2:
                class_ce = F.cross_entropy(
                    z_proj.moveaxis(-1, 1), labels.long(),
                    ignore_index=self.ignore_index,
                    reduction='none',
                )
                ignore_mask = ~(labels == self.ignore_index)
            else:
                # Multiple classes per time step
                targets = labels
                targets[targets == self.ignore_index] = self.proj_size - 1
                targets = F.one_hot(
                    targets.long(), num_classes=self.proj_size
                ).float().sum(dim=-2)
                class_ce = F.binary_cross_entropy_with_logits(
                    z_proj, targets,
                    reduction='none',
                )
                ignore_mask = torch.ones_like(class_ce).float()
                ignore_mask[..., self.proj_size-1] = 0
            class_ce = (class_ce * ignore_mask.float()).sum() / (
                ignore_mask.float().sum() + 1e-5
            )
            summary["losses"]["classification_ce"] = class_ce
        if not self.training:
            with torch.no_grad():
                if self.distance_metric == "cosine":
                    z = z / (torch.norm(z, p=2, dim=-1, keepdim=True) + 1e-8)
                ignore_mask = labels != self.ignore_index
                if labels.ndim == 3:
                    ignore_mask = ignore_mask.any(dim=-1)
                z = z[:, :labels.shape[1]]
                if ignore_mask.sum() > 0:
                    summary["buffers"]["embeddings"] = z[ignore_mask].cpu().numpy()
                    summary["buffers"]["labels"] = labels[ignore_mask].cpu().numpy()
        summary = add_prefix(self.prefix, summary)
        return summary

    def get_posteriors(self, outputs):
        raise NotImplementedError()

    def modify_summary(self, summary):
        if f"{self.prefix}embeddings" in summary['buffers']:
            if self.distance_metric == "cosine":
                dist_fn = cosine
            else:
                raise NotImplementedError(self.distance_metric)
            embeddings = np.concatenate(
                summary['buffers'][f"{self.prefix}embeddings"]
            )
            try:
                _labels = np.concatenate(
                    summary['buffers'][f"{self.prefix}labels"]
                )
            except ValueError:
                _labels = summary['buffers'][f"{self.prefix}labels"]
                # multi-hot labels
                max_num_classes = max(
                    li.shape[1] for li in _labels
                )
                _labels = np.concatenate(list(map(
                    functools.partial(
                        pad_tensor,
                        axis=1,
                        pad=max_num_classes,
                        padding_value=self.ignore_index
                    ), _labels
                )))
            if self.distance_metric == "cosine":
                validation_mean = np.mean(embeddings, axis=0, keepdims=True)
                embeddings = embeddings - validation_mean
            unq_labels = np.unique(_labels)
            unq_labels = unq_labels[unq_labels != self.ignore_index]
            if len(unq_labels) == 1:
                warnings.warn(
                    "Only one class present in labels, EER estimation is not "
                    "meaningful."
                )
                eer = 1.0
                summary["scalars"][f"{self.prefix}EER"] = eer
            else:
                if len(unq_labels) > 20:
                    # Reduce labels to get meaningful EER estimation
                    warnings.warn(
                        "More than 20 classes present in labels, reducing to "
                        "20 randomly for EER estimation."
                    )
                    np.random.default_rng(0).shuffle(unq_labels)
                    unq_labels = unq_labels[:20]
                    label_mask = np.isin(_labels, unq_labels)
                    if _labels.ndim == 2:
                        label_mask = label_mask.any(axis=1)
                    embeddings = embeddings[label_mask]
                    _labels = _labels[label_mask]
                indexer = list(range(len(embeddings)))
                np.random.default_rng(0).shuffle(indexer)
                scores, labels = [], []
                for idx1, idx2 in enumerate(indexer[:6000]):
                    if _labels.ndim == 2:
                        # Multiple classes per time step
                        labels.append(
                            np.isin(
                                _labels[idx1][_labels[idx1]!=self.ignore_index],
                                _labels[idx2][_labels[idx2]!=self.ignore_index]
                            ).any()
                        )
                    else:
                        labels.append(_labels[idx1] == _labels[idx2])
                    scores.append(
                        1 - dist_fn(embeddings[idx1], embeddings[idx2])
                    )
                eer = get_eer(scores, labels)
                summary["scalars"][f"{self.prefix}EER"] = eer
                summary["histograms"][f"{self.prefix}scores"] = np.array(scores)
                summary["histograms"][f"{self.prefix}score_distance_"] = np.abs(
                    np.array(labels) - np.array(scores)
                )
            summary["buffers"].clear()
        return summary
