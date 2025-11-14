from concurrent.futures import ThreadPoolExecutor
from functools import partial
import logging
import os
from pathlib import Path
import typing as tp

import click
from click.exceptions import MissingParameter
from lazy_dataset.core import from_path
import numpy as np
from paderbox.io import load_audio
from paderbox.transform.module_resample import resample_sox
import padertorch as pt
from padertorch.data.utils import collate_fn
import plotext as plt
import psutil
from sed_scores_eval.base_modules.io import write_sed_scores

from .modules.data_loader import LoadAudio, sort_by_length
from .modules.ssl_mos import SpeechQualityPredictor, SAMPLING_RATE
from .utils import nadir_detection

try:
    import dlp_mpi
    DLP_MPI_AVAILABLE = True
except ImportError:
    DLP_MPI_AVAILABLE = False

LOCAL_SQA_INFER_MODEL = os.environ.get('LOCAL_SQA_INFER_MODEL', None)
logger = logging.getLogger("local_sqa.infer")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)


def worker(
    batch: dict,
    model: SpeechQualityPredictor,
    input_path: Path,
    output_dir: Path,
    prominence: bool = False,
    verbose: bool = False,
    **peak_kwargs,
):
    batch_size = batch['audio'].shape[0]
    preds, frame_preds, seq_len_preds = model(
        batch['audio'], batch['num_samples'],
    )
    for audio_path, pred, frame_pred, n_samples in zip(
        batch['audio_path'],
        preds,
        pt.unpad_sequence(
            np.moveaxis(frame_preds, 1, 0), seq_len_preds
        ),
        batch['num_samples'],
    ):
        if verbose:
            logger.info(
                "File: %s - Predicted MOS: %.3f",
                audio_path,
                pred.item(),
            )
        rel_path = audio_path.relative_to(input_path)
        output_path = output_dir / rel_path.with_suffix('.tsv')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        scores = frame_pred
        if prominence:
            prominences, _ = nadir_detection(
                frame_pred, **peak_kwargs
            )
            scores = np.stack([frame_pred, prominences], axis=1)
        timestamps = np.linspace(
            0, n_samples/SAMPLING_RATE,
            num=len(frame_pred)+1, endpoint=True,
        )
        write_sed_scores(
            scores,
            output_path,
            timestamps=timestamps,
            event_classes=["MOS", "Minima prominences"]
        )
    return batch_size


@click.command()
@click.option(
    '--model-dir', '-m', type=click.Path(exists=True, file_okay=False),
)
# TODO: Allow multiple files and directories
@click.option(
    '--input-path', '-i',
    type=click.Path(exists=True, file_okay=True, dir_okay=True),
)
@click.option(
    '-o', '--output-dir', type=click.Path(file_okay=False, dir_okay=True),
)
@click.option(
    '--checkpoint-name', '-c', type=str, default='ckpt_best_SRCC.pth',
)
@click.option(
    '--resample/--no-resample', '-r/-nr', default=False,
)
@click.option(
    '--median-filter-size', '-mf', type=int, default=3,
)
@click.option(
    '--batch-size', '-b', type=int, default=4,
)
@click.option(
    '--prominence', is_flag=True, default=False,
)
@click.option(
    '--peak-rel-height', type=float, default=0.5,
)
@click.option(
    '--peak-width', type=int, default=None,
)
@click.option(
    '--peak-wlen', type=int, default=None,
)
@click.option(
    '--backend', type=click.Choice(['t', 'dlp_mpi']), default='t',
    help=(
        'Backend to use for parallel processing. '
        'Possible choices are threads ("t") and MPI ("dlp_mpi"). '
        'Defaults to "t" (threads).'
    ),
)
@click.option(
    '--num-workers', '-j', type=int, default=-1,
    help=(
        'Number of workers to use for parallel processing. '
        'If -1, uses all available CPU cores.'
    ),
)
@click.option(
    '--verbose', '-v', is_flag=True, default=False,
)
def inference(
    model_dir: tp.Optional[str],
    input_path: tp.Union[str, Path],
    output_dir: tp.Optional[tp.Union[str, Path]],
    checkpoint_name: str = "ckpt_best_SRCC.pth",
    resample: bool = False,
    median_filter_size: tp.Optional[int] = 3,
    batch_size: int = 4,
    prominence: bool = False,
    peak_rel_height: float = 0.5,
    peak_width: tp.Optional[int] = None,
    peak_wlen: tp.Optional[int] = None,
    backend: str = 't',
    num_workers: int = -1,
    verbose: bool = False,
):
    input_path = Path(input_path)

    if model_dir is None:
        if LOCAL_SQA_INFER_MODEL is None:
            raise RuntimeError(
                "Either provide a model to load via --model-dir/-m option or "
                "set the environment variable "
                "LOCAL_SQA_INFER_MODEL to point to a pretrained model "
                "directory."
            )
        model_dir = LOCAL_SQA_INFER_MODEL
    logger.info("Loading SQA model from %s", model_dir)
    model = SpeechQualityPredictor(
        storage_dir=model_dir,
        checkpoint_name=checkpoint_name,
        return_numpy=True,
        prepare_example_before=input_path.is_file(),
        median_filter_size=median_filter_size,
        consider_mpi=backend=='dlp_mpi',
    )

    if peak_wlen is None:
        # Defaults to 1 second window
        peak_wlen = model.model.encoder.frame_rate

    # File input
    if input_path.is_file():
        wav, sr = load_audio(input_path, return_sample_rate=True)
        if sr != SAMPLING_RATE and resample:
            wav = resample_sox(wav, in_rate=sr, out_rate=SAMPLING_RATE)
        elif sr != SAMPLING_RATE:
            raise RuntimeError(
                f"Input audio sampling rate {sr} does not match the model's "
                f"sampling rate of {SAMPLING_RATE}Hz. Use --resample to "
                "resample the audio."
            )
        while wav.ndim < 3:
            wav = wav[None]

        logger.info(
            "Analysing file: %s\n\tFile length: %.3f s",
            input_path,
            wav.shape[-1] / SAMPLING_RATE,
        )

        preds, frame_preds, seq_len_preds = model(wav, [wav.shape[-1]])

        logger.info(
            "Overall predicted MOS: %.3f",
            preds[0].item()
        )
        # Plot frame-level predictions for a single file
        t_axis = np.arange(seq_len_preds[0]) / model.model.encoder.frame_rate

        if prominence:
            plt.subplots(2, 1)
            plt.subplot(1, 1)  # Activate first subplot

        frame_preds = frame_preds[0]
        plt.plot(t_axis, frame_preds)
        plt.ylabel("Predicted MOS")
        plt.grid(1, 1)

        if prominence:
            # Plot nadir prominences
            prominences, _ = nadir_detection(
                frame_preds,
                width=peak_width,
                rel_height=peak_rel_height,
                wlen=peak_wlen,
            )
            plt.subplot(2, 1)  # Activate second subplot
            plt.plot(t_axis, prominences)
            plt.ylabel("Prominence of local minima")
            plt.grid(1, 1)

        plt.xlabel("Time / s")
        plt.show()
        return

    # Directory input
    if input_path.is_dir():
        if output_dir is None:
            raise MissingParameter(
                "Output directory must be specified when input is a directory.",
                param=click.Option(['--output-dir', '-o']),
            )
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Analysing all audio files in directory: %s", input_path)
        examples = from_path(input_path, suffix=['.wav', '.flac'])
        num_examples = len(examples)
        logger.info("Found %d audio files.", num_examples)
        examples = (
            examples
            .map(LoadAudio(
                audio_path_keys=['wav', 'flac'],
                target_sampling_rate=SAMPLING_RATE,
                resample=resample,
            ))
            .map(model.model.prepare_example)
            .batch(batch_size)
            .map(sort_by_length)
            .map(collate_fn)
            .map(partial(model.model.example_to_device, device=model.device))
        )
        if prominence:
            logger.info(
                "--prominence flag set. Writing local minima prominences."
            )
        logger.info(
            "Writing results to output directory: %s", output_dir
        )
        logger.info("Processing batches of size %d.", batch_size)

        _worker = partial(
            worker,
            model=model,
            input_path=input_path,
            output_dir=output_dir,
            prominence=prominence,
            verbose=verbose,
            width=peak_width,
            rel_height=peak_rel_height,
            wlen=peak_wlen,
        )
        if backend == 't':
            if num_workers < 0:
                num_workers = len(psutil.Process().cpu_affinity())
            logger.info(
                "Using threads backend with %d workers", num_workers
            )
            executor = ThreadPoolExecutor(max_workers=num_workers).map(
                _worker, examples
            )
        else:
            if not DLP_MPI_AVAILABLE:
                raise ImportError(
                    "dlp_mpi is not available. "
                    "Please install it to use the 'dlp_mpi' backend.\n"
                    "See: https://github.com/fgnt/dlp_mpi"
                )
            logger.info("Using MPI backend")
            executor = dlp_mpi.map_unordered(
                _worker, examples,
                indexable=False,
                progress_bar=True,
            )
        for _ in executor:
            pass
        logger.info("Finished.")


if __name__ == '__main__':
    inference()  # pylint: disable=no-value-for-parameter
