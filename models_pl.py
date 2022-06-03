from abc import ABC
import argparse
import os
from argparse import Namespace
from typing import Dict, Any, Tuple
import numpy as np

import torch
from torch import nn
import torch.nn.parallel
import torch.utils.data
import pytorch_lightning as pl
import torchvision
from rtpt import RTPT

from torch.nn import functional as F
from args import parse_args
from simple_einet.data import get_data_shape, Dist, get_distribution
from exp_utils import (
    setup_experiment_lit,
    load_from_checkpoint,
)
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger

from simple_einet.data import build_dataloader
from simple_einet.einet import EinetConfig, Einet
from simple_einet.distributions.binomial import Binomial


# Translate the dataloader index to the dataset name
DATALOADER_ID_TO_SET_NAME = {0: "val", 1: "test"}


def make_einet(args, num_classes: int = 1):
    """
    Make an EinsumNetworks model based off the given arguments.

    Args:
        args: Arguments parsed from argparse.

    Returns:
        EinsumNetworks model.
    """
    image_shape = get_data_shape(args.dataset)
    # leaf_kwargs, leaf_type = {"total_count": 255}, Binomial
    leaf_kwargs, leaf_type = get_distribution(
        dist=args.dist, min_sigma=args.min_sigma, max_sigma=args.max_sigma
    )

    config = EinetConfig(
        num_features=image_shape.num_pixels,
        num_channels=image_shape.channels,
        depth=args.D,
        num_sums=args.S,
        num_leaves=args.I,
        num_repetitions=args.R,
        num_classes=num_classes,
        leaf_kwargs=leaf_kwargs,
        leaf_type=leaf_type,
        dropout=args.dropout,
        cross_product=args.cp,
    )
    return Einet(config)


class LitModel(pl.LightningModule, ABC):
    def __init__(self, args: argparse.Namespace, name: str) -> None:
        super().__init__()
        self.args = args
        self.image_shape = get_data_shape(args.dataset)
        self.rtpt = RTPT(
            name_initials="SL",
            experiment_name="einet_" + name + ("_" + str(args.tag) if args.tag else ""),
            max_iterations=args.epochs + 1,
        )
        self.save_hyperparameters()

    def preprocess(self, data: torch.Tensor):
        if self.args.dist == Dist.BINOMIAL:
            data *= 255.0

        return data

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(0.7 * self.args.epochs), int(0.9 * self.args.epochs)],
            gamma=0.1,
        )
        return [optimizer], [lr_scheduler]

    def on_train_start(self) -> None:
        self.rtpt.start()


class SpnGenerative(LitModel):
    def __init__(self, args: Namespace):
        super().__init__(args=args, name="gen")
        self.spn = make_einet(args)

    def training_step(self, train_batch, batch_idx):
        data, labels = train_batch
        data = self.preprocess(data)
        nll = self.negative_log_likelihood(data)
        self.log("Train/nll", nll)
        return nll

    def validation_step(self, val_batch, batch_idx):
        data, labels = val_batch
        data = self.preprocess(data)
        nll = self.negative_log_likelihood(data)
        self.log("Val/nll", nll)
        return nll

    def negative_log_likelihood(self, data):
        """
        Compute negative log likelihood of data.

        Args:
            data: Data to compute negative log likelihood of.

        Returns:
            Negative log likelihood of data.
        """
        nll = -1 * self.spn(data).mean()
        return nll

    def generate_samples(self, num_samples: int):
        samples = self.spn.sample(num_samples=num_samples, mpe_at_leaves=True).view(
            -1, *self.image_shape
        )
        samples = samples / 255.0
        return samples

    def on_train_epoch_end(self):

        with torch.no_grad():
            samples = self.generate_samples(num_samples=25)
            grid = torchvision.utils.make_grid(
                samples.data[:25], nrow=5, pad_value=0.0, normalize=True
            )
            self.logger.log_image(key="samples", images=[grid])

        super().on_train_epoch_end()

    def save_samples(self, samples_dir, num_samples, nrow):
        """
        Save samples to a directory.

        Args:
            samples_dir: Directory to save samples to.
            num_samples: Number of samples to save.
            nrow: Number of samples per row.

        """
        for i in range(5):
            samples = self.generate_samples(num_samples)
            grid = torchvision.utils.make_grid(
                samples.data[:25], nrow=nrow, pad_value=0.0, normalize=True
            )
            torchvision.utils.save_image(grid, os.path.join(samples_dir, f"{i}.png"))

    def test_step(self, batch, batch_idx, dataloader_id=0):
        data, labels = batch

        data = self.preprocess(data)
        nll = self.negative_log_likelihood(data)

        set_name = DATALOADER_ID_TO_SET_NAME[dataloader_id]
        self.log(f"Test/{set_name}_nll", nll, add_dataloader_idx=False)


class SpnDiscriminative(LitModel):
    """
    Discriminative SPN model. Models the class conditional data distribution at its C root nodes.
    """
    def __init__(self, args: Namespace):
        super().__init__(args, name="disc")

        # Construct SPN
        self.spn = make_einet(args, num_classes=10)

        # Define loss function
        self.criterion = nn.CrossEntropyLoss()

    def training_step(self, train_batch, batch_idx):
        loss, accuracy = self._get_cross_entropy_and_accuracy(train_batch)
        self.log("Train/accuracy", accuracy, prog_bar=True)
        self.log("Train/cross_entropy", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss, accuracy = self._get_cross_entropy_and_accuracy(val_batch)
        self.log("Val/cross_entropy", loss)
        self.log("Val/accuracy", accuracy)
        return loss

    def test_step(self, batch, batch_idx, dataloader_id=0):
        loss, accuracy = self._get_cross_entropy_and_accuracy(batch)
        set_name = DATALOADER_ID_TO_SET_NAME[dataloader_id]
        self.log(f"Test/{set_name}_cross_entropy", loss, add_dataloader_idx=False)
        self.log(f"Test/{set_name}_accuracy", accuracy, add_dataloader_idx=False)

    def _get_cross_entropy_and_accuracy(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute cross entropy loss and accuracy of batch.
        Args:
            batch: Batch of data.

        Returns:
            Tuple of (cross entropy loss, accuracy).
        """
        data, labels = batch
        data = self.preprocess(data)
        outputs = self.spn(data)  # [N, C]
        loss = self.criterion(outputs, labels)
        accuracy = (labels == outputs.argmax(-1)).sum() / outputs.shape[0]
        return loss, accuracy