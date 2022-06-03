#!/usr/bin/env python
from pytorch_lightning.utilities.model_summary import ModelSummary, get_human_readable_count
import os
from argparse import Namespace
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn.parallel
import torch.utils.data
import pytorch_lightning as pl
import torchvision
from rtpt import RTPT

from args import parse_args
from models_pl import SpnDiscriminative, SpnGenerative
from simple_einet.data import get_data_shape, Dist
from exp_utils import (
    setup_experiment_lit,
    load_from_checkpoint,
)
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from simple_einet.data import build_dataloader
from simple_einet.einet import EinetConfig, Einet
from simple_einet.distributions.binomial import Binomial


def main(args, model: SpnGenerative, results_dir, hparams: Dict[str, Any]):
    seed_everything(args.seed, workers=True)

    print("Training model...")
    # Create dataloader
    normalize = args.dist == Dist.NORMAL
    train_loader, val_loader, test_loader = build_dataloader(
        args=args, loop=False, normalize=normalize
    )

    # Create callbacks
    # logger_tb = TensorBoardLogger(results_dir, name="tb")
    logger_wandb = WandbLogger(name=args.tag, project="einet")

    # Store number of model parameters
    summary = ModelSummary(model, max_depth=-1)
    print(summary)
    logger_wandb.experiment.config["trainable_parameters"] = summary.trainable_parameters
    logger_wandb.experiment.config["trainable_parameters_leaf"] = summary.param_nums[
        summary.layer_names.index("spn.leaf")
    ]
    logger_wandb.experiment.config["trainable_parameters_sums"] = summary.param_nums[
        summary.layer_names.index("spn.einsum_layers")
    ]

    # Create trainer
    gpus = 1 if torch.cuda.is_available() else 0
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    # if torch.backends.mps.is_available():  # MPS not supported by PL for now
    #     accelerator = "mps"
    devices = [args.gpu] if gpus else None
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        logger=logger_wandb,
        accelerator=accelerator,
        devices=devices,
        # val_check_interval=0.2,
    )

    if not args.load_and_eval:
        # Fit model
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print("Evaluating model...")

    # Evaluate spn reconstruction error
    test_val = trainer.test(model=model, dataloaders=[train_loader, val_loader, test_loader], verbose=True)

    # Save some samples
    if isinstance(model, SpnGenerative):
        samples_dir = os.path.join(results_dir, "samples")
        os.makedirs(exist_ok=True, name=samples_dir)
        model.save_samples(samples_dir, num_samples=25, nrow=5)


if __name__ == "__main__":
    args = parse_args()
    results_dir, args = setup_experiment_lit(name="spn", args=args, remove_if_exists=True)

    # Load or create model
    if args.load_and_eval:
        model = load_from_checkpoint(
            results_dir, load_fn=SpnGenerative.load_from_checkpoint, args=args
        )
    else:
        if args.classification:
            model = SpnDiscriminative(args)
        else:
            model = SpnGenerative(args)

    hparams = {
        "lr": args.lr,
    }
    main(
        args=args,
        model=model,
        results_dir=results_dir,
        hparams=hparams,
    )
