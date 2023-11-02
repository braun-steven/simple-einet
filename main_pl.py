#!/usr/bin/env python
import omegaconf
import time
import wandb
from hydra.core.hydra_config import HydraConfig
import logging
from omegaconf import DictConfig, OmegaConf, open_dict
import os
from rich.traceback import install

install()

import hydra
import pytorch_lightning as pl
import torch.utils.data
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import StochasticWeightAveraging, RichProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.model_summary import (
    ModelSummary,
)

from exp_utils import (
    load_from_checkpoint,
    plot_distribution,
)
from models_pl import SpnDiscriminative, SpnGenerative
from simple_einet.data import Dist
from simple_einet.data import build_dataloader
from simple_einet.sampling_utils import init_einet_stats

# A logger for this file
logger = logging.getLogger(__name__)


def main(cfg: DictConfig):
    """
    Main function for training and evaluating an Einet.

    Args:
        cfg: Config file.
    """
    preprocess_cfg(cfg)

    # Get hydra config
    hydra_cfg = HydraConfig.get()
    run_dir = hydra_cfg.runtime.output_dir
    logger.info("Working directory : {}".format(os.getcwd()))

    # Save config
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        OmegaConf.save(config=cfg, f=f)

    # Safe run_dir in config (use open_dict to make config writable)
    with open_dict(cfg):
        cfg.run_dir = run_dir

    logger.info("\n" + OmegaConf.to_yaml(cfg, resolve=True))
    logger.info("Run dir: " + run_dir)

    seed_everything(cfg.seed, workers=True)

    if not cfg.wandb:
        os.environ["WANDB_MODE"] = "offline"

    # Ensure that everything is properly seeded
    seed_everything(cfg.seed, workers=True)

    # Setup devices
    if torch.cuda.is_available():
        accelerator = "gpu"
        if type(cfg.gpu) == int:
            devices = [int(cfg.gpu)]
        else:
            devices = [int(g) for g in cfg.gpu]
    else:
        accelerator = "cpu"
        devices = 1

    logger.info("Training model...")
    # Create dataloader
    normalize = cfg.dist in [Dist.NORMAL, Dist.NORMAL_RAT, Dist.MULTIVARIATE_NORMAL]
    train_loader, val_loader, test_loader = build_dataloader(
        dataset_name=cfg.dataset,
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        num_workers=min(cfg.num_workers, os.cpu_count()),
        loop=False,
        normalize=normalize,
    )

    # Create callbacks
    cfg_container = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    logger_wandb = WandbLogger(
        name=cfg.tag,
        project=cfg.project_name,
        group=cfg.group_tag,
        offline=not cfg.wandb,
        config=cfg_container,
        reinit=True,
        save_dir=run_dir,
        settings=wandb.Settings(start_method="thread"),
    )

    # Load or create model
    if cfg.load_and_eval:
        model = load_from_checkpoint(
            run_dir,
            load_fn=SpnGenerative.load_from_checkpoint,
            args=cfg,
        )
    else:
        if cfg.classification:
            model = SpnDiscriminative(cfg, steps_per_epoch=len(train_loader))
        else:
            model = SpnGenerative(cfg, steps_per_epoch=len(train_loader))

        if cfg.torch_compile:  # Doesn't seem to work with einsum yet
            # Rase an error since einsum doesn't seem to work with compilation yet
            # model = torch.compile(model)
            raise NotImplementedError("Torch compilation not yet supported with einsum.")

        if cfg.einet_mixture:
            # If we chose a mixture of einets, we need to initialize the mixture weights
            logger.info("Initializing Einet mixture weights")
            model.spn.initialize(dataloader=train_loader, device=devices[0])

        if cfg.init_leaf_data:
            logger.info("Initializing leaf distributions from data statistics")
            init_einet_stats(model.spn, train_loader)

    # Store number of model parameters
    summary = ModelSummary(model, max_depth=-1)
    logger.info("Model:")
    logger.info(model)
    logger.info("Summary:")
    logger.info(summary)

    # Setup callbacks
    callbacks = []

    # Add StochasticWeightAveraging callback
    if cfg.swa:
        swa_callback = StochasticWeightAveraging()
        callbacks.append(swa_callback)

    # Enable rich progress bar
    if not cfg.debug:
        # Cannot "breakpoint()" in the training loop when RichProgressBar is active
        callbacks.append(RichProgressBar())

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        logger=logger_wandb,
        accelerator=accelerator,
        devices=devices,
        callbacks=callbacks,
        precision=cfg.precision,
        fast_dev_run=cfg.debug,
        profiler=cfg.profiler,
        default_root_dir=run_dir,
        enable_checkpointing=False,
        detect_anomaly=True,
    )

    if not cfg.load_and_eval:
        # Fit model
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    logger.info("Evaluating model...")

    if "synth" in cfg.dataset and not cfg.classification:
        plot_distribution(model=model.spn, dataset_name=cfg.dataset, logger_wandb=logger_wandb)

    # Evaluate spn reconstruction error
    trainer.test(model=model, dataloaders=[train_loader, val_loader, test_loader], verbose=True)
    logger.info("Finished evaluation...")

    # Save checkpoint in general models directory to be used across experiments
    chpt_path = os.path.join(run_dir, "model.pt")
    logger.info("Saving checkpoint: " + chpt_path)
    trainer.save_checkpoint(chpt_path)


def preprocess_cfg(cfg: DictConfig):
    """
    Preprocesses the config file.
    Replace defaults if not set (such as data/results dir).

    Args:
        cfg: Config file.
    """
    home = os.getenv("HOME")

    # If results dir is not set, get from ENV, else take ~/data
    if "data_dir" not in cfg:
        cfg.data_dir = os.getenv("DATA_DIR", os.path.join(home, "data"))

    # If results dir is not set, get from ENV, else take ~/results
    if "results_dir" not in cfg:
        cfg.results_dir = os.getenv("RESULTS_DIR", os.path.join(home, "results"))

    # If FP16/FP32 is given, convert to int (else it's "bf16", keep string)
    if cfg.precision == "16" or cfg.precision == "32":
        cfg.precision = int(cfg.precision)

    if "profiler" not in cfg:
        cfg.profiler = None  # Accepted by PyTorch Lightning Trainer class

    if "tag" not in cfg:
        cfg.tag = None

    if "group_tag" not in cfg:
        cfg.group_tag = None

    if "seed" not in cfg:
        cfg.seed = int(time.time())

    # Convert dist string to enum
    cfg.dist = Dist[cfg.dist.upper()]


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main_hydra(cfg: DictConfig):
    try:
        main(cfg)
    except Exception as e:
        logging.critical(e, exc_info=True)  # log exception info at CRITICAL log level
    finally:
        # Close wandb instance. Necessary for hydra multi-runs where main() is called multipel times
        wandb.finish()


if __name__ == "__main__":
    main_hydra()
