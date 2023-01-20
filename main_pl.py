#!/usr/bin/env python
from simple_einet.distributions.normal import Normal
from simple_einet.einet import Einet
import omegaconf
import time
import wandb
from hydra.core.hydra_config import HydraConfig
import logging
from omegaconf import DictConfig, OmegaConf, open_dict
import os
import sys
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
    setup_experiment,
    load_from_checkpoint,
    plot_distribution,
)
from models_pl import SpnDiscriminative, SpnGenerative
from simple_einet.data import Dist
from simple_einet.data import build_dataloader
from tqdm import tqdm

# A logger for this file
logger = logging.getLogger(__name__)


def init_einet_stats(einet: Einet, dataloader: torch.utils.data.DataLoader):
    stats_mean = None
    stats_std = None
    for batch in tqdm(dataloader, desc="Leaf Parameter Initialization"):
        data, label = batch
        if stats_mean == None:
            stats_mean = data.mean(dim=0)
            stats_std = data.std(dim=0)
        else:
            stats_mean += data.mean(dim=0)
            stats_std += data.std(dim=0)

    stats_mean /= len(dataloader)
    stats_std /= len(dataloader)

    if einet.config.leaf_type == Normal:
        if type(einet) == Einet:
            einets = [einet]
        else:
            einets = einet.einets

        stats_mean_v = (
            stats_mean.view(-1, 1, 1)
            .repeat(1, einets[0].config.num_leaves, einets[0].config.num_repetitions)
            .view_as(einets[0].leaf.base_leaf.means)
        )
        stats_std_v = (
            stats_std.view(-1, 1, 1)
            .repeat(1, einets[0].config.num_leaves, einets[0].config.num_repetitions)
            .view_as(einets[0].leaf.base_leaf.log_stds)
        )
        for net in einets:
            net.leaf.base_leaf.means.data = stats_mean_v

            net.leaf.base_leaf.log_stds.data = torch.log(stats_std_v + 1e-3)


def main(cfg: DictConfig):
    preprocess_cfg(cfg)

    hydra_cfg = HydraConfig.get()
    run_dir = hydra_cfg.runtime.output_dir
    print("Working directory : {}".format(os.getcwd()))

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
        devices = None

    print("Training model...")
    # Create dataloader
    normalize = cfg.dist == Dist.NORMAL
    train_loader, val_loader, test_loader = build_dataloader(cfg=cfg, loop=False, normalize=normalize)

    # Create callbacks
    cfg_container = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    logger_wandb = WandbLogger(
        name=cfg.tag,
        project=cfg.project_name,
        group=cfg.group_tag,
        offline=not cfg.wandb,
        config=cfg_container,
        reinit=False,
        save_dir=run_dir,
        settings=wandb.Settings(start_method="thread"),
    )

    # Load or create model
    if cfg.load_and_eval:
        model = load_from_checkpoint(
            run_dir,
            load_fn=SpnGenerative.load_from_checkpoint,
            cfg=cfg,
        )
    else:
        if cfg.classification:
            model = SpnDiscriminative(cfg, steps_per_epoch=len(train_loader))
        else:
            model = SpnGenerative(cfg, steps_per_epoch=len(train_loader))

        if cfg.einet_mixture:
            model.spn.initialize(dataloader=train_loader, device=devices[0])

        if cfg.init_leaf_data:
            logger.info("Initializing leaf distributions from data statistics")
            init_einet_stats(model.spn, train_loader)

    # Store number of model parameters
    summary = ModelSummary(model, max_depth=-1)
    print("Model:")
    print(model)
    print("Summary:")
    print(summary)
    # logger_wandb.experiment.config["trainable_parameters"] = summary.trainable_parameters
    # logger_wandb.experiment.config["trainable_parameters_leaf"] = summary.param_nums[
    #     summary.layer_names.index("spn.leaf")
    # ]
    # logger_wandb.experiment.config["trainable_parameters_sums"] = summary.param_nums[
    #     summary.layer_names.index("spn.einsum_layers")
    # ]


    # Setup callbacks
    callbacks = []

    # Add StochasticWeightAveraging callback
    if cfg.swa:
        swa_callback = StochasticWeightAveraging()
        callbacks.append(swa_callback)

    # Enable rich progress bar
    callbacks.append(RichProgressBar())

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        # max_steps=cfg.max_steps,
        logger=logger_wandb,
        accelerator=accelerator,
        devices=devices,
        callbacks=callbacks,
        precision=cfg.precision,
        fast_dev_run=cfg.debug,
        profiler=cfg.profiler,
        default_root_dir=run_dir,
        enable_checkpointing=False,
    )

    if not cfg.load_and_eval:
        # Fit model
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print("Evaluating model...")

    if "synth" in cfg.dataset and not cfg.classification:
        plot_distribution(model=model.spn, dataset_name=cfg.dataset, logger_wandb=logger_wandb)

    # Evaluate spn reconstruction error
    trainer.test(model=model, dataloaders=[train_loader, val_loader, test_loader], verbose=True)

    print("Finished evaluation...")

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
        cfg.env.seed = int(time.time())

    if cfg.K > 0:
        cfg.I = cfg.K
        cfg.S = cfg.K

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
