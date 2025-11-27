from itertools import islice
import logging
from pathlib import Path
import random
import shutil
import tempfile

import hydra
from omegaconf import DictConfig, OmegaConf

import padertorch as pt
from padertorch.train.hooks import LRAnnealingHook
import torch
from tqdm.auto import tqdm

from .modules.data_loader import JsonParser, Dataloader

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="default")
def main(config: DictConfig):
    OmegaConf.resolve(config)

    _config = OmegaConf.to_container(config, resolve=True)
    _config['trainer'] = pt.Trainer.get_config(_config['trainer'])

    if config.trainer.storage_dir is None:
        # Create new storage directory
        if config.launch.dry_run:
            storage_dir = Path(tempfile.mkdtemp())
        else:
            storage_dir = pt.io.get_new_subdir(
                basedir=Path(config.base_dir),
            )
            _config["trainer"]["storage_dir"] = str(storage_dir)
        config.trainer.storage_dir = str(storage_dir)
    config_file = Path(config.trainer.storage_dir) / 'config.yaml'
    if config.launch.train and not config_file.exists():
        pt.io.dump_config(_config, config_file)

    if config.launch.resume:
        if config.launch.dry_run:
            raise RuntimeError("Cannot resume in dry run mode.")
        config = OmegaConf.load(
            Path(config.trainer.storage_dir) / 'config.yaml'
        )
        config.launch.resume = True

    log.info(OmegaConf.to_yaml(config))

    parsers = []
    for _, db_conf in config.databases.items():
        db_conf = OmegaConf.to_container(db_conf, resolve=True)
        parsers.append(JsonParser.from_config(db_conf))

    trainer_config = OmegaConf.to_container(config.trainer, resolve=True)
    trainer_config = pt.Trainer.get_config(trainer_config)
    trainer = pt.Trainer.from_config(trainer_config)

    train_dataloader = Dataloader.from_config(
        OmegaConf.to_container(config.train_dataloader, resolve=True)
    )(
        *parsers,
        prepare_example_fn=getattr(trainer.model, "prepare_example", None),
    )
    val_dataloader = Dataloader.from_config(
        OmegaConf.to_container(config.val_dataloader, resolve=True)
    )(
        *parsers,
        prepare_example_fn=getattr(trainer.model, "prepare_example", None),
    )

    if config.launch.accelerator == "auto":
        device = 0 if torch.cuda.is_available() else "cpu"
    else:
        device = config.launch.accelerator

    decay_factor = config.min_lr/config.lr
    trainer.register_hook(
        LRAnnealingHook(
            [1, config.trainer.stop_trigger[1]],
            [(config.trainer.stop_trigger[0], decay_factor)],
            config.trainer.stop_trigger[1],
        ),
    )

    if config.launch.test_run:
        trainer.test_run(train_dataloader, val_dataloader, device=device)
        # Test if model can be loaded
        pt.io.dump_config(
            _config, Path(trainer.storage_dir) / 'config.yaml'
        )
        (trainer.storage_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint()
        pt.Model.from_storage_dir(
            trainer.storage_dir, config_name="config.yaml",
            checkpoint_name="ckpt_latest.pth"
        )

    if config.launch.train:
        if (
            not config.launch.resume
            and hasattr(trainer.model, "test_seed")
            and config.num_seeds > 0
        ):
            # Test seeds
            log.info("Testing seeds.")
            best_metric = None
            best_seed = None
            seed = torch.initial_seed()
            if isinstance(device, list):
                _device = device[0]
            else:
                _device = device
            trainer.model.to(_device).eval()
            subiterator = list(islice(
                iter(val_dataloader),
                int(
                    config.sub_iterator_length
                    / config.val_dataloader.batch_size
                )
            ))
            for _ in tqdm(range(config.num_seeds), desc="Testing seeds"):
                summary = trainer.model.test_seed(subiterator, seed, device)
                metric = summary["scalars"][config.validation.metric]
                if (
                    best_metric is None
                    or (config.validation.maximize and metric > best_metric)
                    or (not config.validation.maximize and metric < best_metric)
                ):
                    best_metric = metric
                    best_seed = seed
                    log.debug(
                        "New best %s: %f", config.validation.metric, best_metric
                    )
                del summary
                seed = torch.seed()

            del subiterator
            if best_seed is not None:
                torch.manual_seed(best_seed)
                torch.cuda.manual_seed(best_seed)
                random.seed(best_seed)
                trainer.model.reset_parameters(best_seed)
                log.info(
                    "Finished testing seeds. Best %s: %f",
                    config.validation.metric, best_metric
                )
        if not config.launch.resume:
            try:
                if trainer.model.bilstm.zero_init_():
                    log.info(
                        "zero init: setting Transformer MLP weights to zeros."
                    )
            except AttributeError:
                pass
            try:
                if trainer.model.comparator.zero_init_():
                    log.info(
                        "zero init: setting Transformer MLP weights to zeros."
                    )
            except AttributeError:
                pass
            if config.zero_init:
                trainer.model.zero_init_()
                log.info(
                    "zero init: setting output layer weights to zeros."
                )

        if val_dataloader is not None:
            log.info("Registering validation hook.")
            trainer.register_validation_hook(
                val_dataloader, metric=config.validation.metric,
                lr_update_factor=config.validation.lr_update_factor,
                n_back_off=config.validation.n_back_off,
                back_off_patience=config.validation.back_off_patience,
                maximize=config.validation.maximize,
            )
        trainer.train(
            train_dataloader, resume=config.launch.resume,
            device=device,
        )

    if config.launch.dry_run:
        log.info("Dry run: Removing storage directory %s.", trainer.storage_dir)
        shutil.rmtree(config.trainer.storage_dir)


if __name__ == "__main__":
    main()
