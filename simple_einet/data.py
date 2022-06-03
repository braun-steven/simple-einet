import itertools
import random
import os
import math
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import Sampler
from torchvision.datasets import CIFAR10, MNIST, SVHN, CelebA, FashionMNIST, LSUN
from torchvision.transforms.functional import InterpolationMode

from simple_einet.distributions import RatNormal
from simple_einet.distributions.binomial import Binomial


@dataclass
class Shape:
    channels: int  # Number of channels
    height: int  # Height in pixels
    width: int  # Width in pixels

    def __iter__(self):
        for element in [self.channels, self.height, self.width]:
            yield element

    def __getitem__(self, index: int):
        return [self.channels, self.height, self.width][index]

    def downscale(self, scale):
        """Downscale this shape by the given scale. Only changes height/width."""
        return Shape(self.channels, round(self.height / scale), round(self.width / scale))

    def upscale(self, scale):
        """Upscale this shape by the given scale. Only changes height/width."""
        return Shape(self.channels, round(self.height * scale), round(self.width * scale))

    @property
    def num_pixels(self):
        return self.width * self.height



def get_data_shape(dataset_name: str) -> Shape:
    """Get the expected data shape.

    Args:
        dataset_name (str): Dataset name.

    Returns:
        Tuple[int, int, int]: Tuple of [channels, height, width].
    """
    return Shape(
        *{
            "mnist": (1, 32, 32),
            "mnist-28": (1, 28, 28),
            "fmnist": (1, 32, 32),
            "fmnist-28": (1, 28, 28),
            "cifar": (3, 32, 32),
            "svhn": (3, 32, 32),
            "celeba": (3, 64, 64),
            "celeba-small": (3, 64, 64),
            "celeba-tiny": (3, 32, 32),
        }[dataset_name]
    )


def get_datasets(args, normalize: bool) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Get the specified dataset.

    Args:
      args: Args.
      normalize: Normalize the dataset.

    Returns:
        Dataset: Dataset.
    """

    # Get dataset name
    dataset_name: str = args.dataset

    # Get the image size (assumes quadratic images)
    shape = get_data_shape(dataset_name)

    # Compose image transformations
    transform = transforms.Compose(
        [
            transforms.Resize(
                size=(shape.height, shape.width),
            ),
            transforms.ToTensor(),
        ]
    )

    kwargs = dict(root=args.data_dir, download=True, transform=transform)

    # Select the datasets
    if dataset_name == "mnist" or dataset_name == "mnist-28":
        if normalize:
            transform.transforms.append(transforms.Normalize([0.5], [0.5]))

        dataset_train = MNIST(**kwargs, train=True)

        dataset_test = MNIST(**kwargs, train=False)

        # for dataset in [dataset_train, dataset_test]:
        #     digits = [0, 1]
        #     mask = torch.zeros_like(dataset.targets).bool()
        #     for digit in digits:
        #         mask = mask | (dataset.targets == digit)

        #     dataset.data = dataset.data[mask]
        #     dataset.targets = dataset.targets[mask]

        N = len(dataset_train.data)
        N_train = round(N * 0.9)
        N_val = N - N_train
        lenghts = [N_train, N_val]
        dataset_train, dataset_val = random_split(dataset_train, lengths=lenghts)

    elif dataset_name == "fmnist" or dataset_name == "fmnist-28":
        if normalize:
            transform.transforms.append(transforms.Normalize([0.5], [0.5]))

        dataset_train = FashionMNIST(**kwargs, train=True)

        dataset_test = FashionMNIST(**kwargs, train=False)

        N = len(dataset_train.data)
        N_train = round(N * 0.9)
        N_val = N - N_train
        lenghts = [N_train, N_val]
        dataset_train, dataset_val = random_split(dataset_train, lengths=lenghts)

    elif "celeba" in dataset_name:
        if normalize:
            transform.transforms.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

        dataset_train = CelebA(**kwargs, split="train")
        dataset_val = CelebA(**kwargs, split="valid")
        dataset_test = CelebA(**kwargs, split="test")

    elif dataset_name == "cifar":
        if normalize:
            transform.transforms.append(transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
        dataset_train = CIFAR10(**kwargs, train=True)

        N = len(dataset_train.data)
        N_train = round(N * 0.9)
        N_val = N - N_train
        lenghts = [N_train, N_val]
        dataset_train, dataset_val = random_split(dataset_train, lengths=lenghts)
        dataset_test = CIFAR10(**kwargs, train=False)

    elif dataset_name == "svhn":
        if normalize:
            transform.transforms.append(transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))

        dataset_train = SVHN(**kwargs, split="train")

        N = len(dataset_train.data)
        lenghts = [round(N * 0.9), round(N * 0.1)]
        dataset_train, dataset_val = random_split(dataset_train, lengths=lenghts)
        dataset_test = SVHN(**kwargs, split="test")

    else:
        raise Exception(f"Unknown dataset: {dataset_name}")

    return dataset_train, dataset_val, dataset_test


def build_dataloader(
    args, loop: bool, normalize: bool
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # Get dataset objects
    dataset_train, dataset_val, dataset_test = get_datasets(args, normalize=normalize)

    # Build data loader
    loader_train = _make_loader(args, dataset_train, loop=loop, shuffle=True)
    loader_val = _make_loader(args, dataset_val, loop=loop, shuffle=False)
    loader_test = _make_loader(args, dataset_test, loop=loop, shuffle=False)
    return loader_train, loader_val, loader_test


def _make_loader(args, dataset: Dataset, loop: bool, shuffle: bool) -> DataLoader:
    if loop:
        sampler = TrainingSampler(size=len(dataset))
    else:
        sampler = None

    from exp_utils import worker_init_reset_seed

    return DataLoader(
        dataset,
        shuffle=(sampler is None) and shuffle,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=worker_init_reset_seed,
    )


class TrainingSampler(Sampler):
    """
    In training, we only care about the "infinite stream" of training data.
    So this sampler produces an infinite stream of indices and
    all workers cooperate to correctly shuffle the indices and sample different indices.

    The samplers in each worker effectively produces `indices[worker_id::num_workers]`
    where `indices` is an infinite stream of indices consisting of
    `shuffle(range(size)) + shuffle(range(size)) + ...` (if shuffle is True)
    or `range(size) + range(size) + ...` (if shuffle is False)
    """

    def __init__(self, size: int, shuffle: bool = True, seed: Optional[int] = None):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        """
        self._size = size
        assert size > 0
        self._shuffle = shuffle
        if seed is None:
            seed = 0
        self._seed = int(seed)

        self._rank = 0
        self._world_size = 1

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            if self._shuffle:
                yield from torch.randperm(self._size, generator=g).tolist()
            else:
                yield from torch.arange(self._size).tolist()


class Dist(str, Enum):
    """Enum for the distribution of the data."""

    NORMAL = "normal"
    BINOMIAL = "binomial"


def get_distribution(dist, min_sigma, max_sigma):
    """
    Get the distribution for the leaves.

    Args:
        dist: The distribution to use.
        min_sigma: The minimum sigma for the leaves.
        max_sigma: The maximum sigma for the leaves.

    Returns:
        leaf_type: The type of the leaves.
        leaf_kwargs: The kwargs for the leaves.

    """
    if dist == Dist.NORMAL:
        leaf_type = RatNormal
        leaf_kwargs = {"min_sigma": min_sigma, "max_sigma": max_sigma}
    elif dist == Dist.BINOMIAL:
        leaf_type = Binomial
        leaf_kwargs = {"total_count": 2**8 - 1}
    else:
        raise ValueError("dist must be either normal or binomial")
    return leaf_kwargs, leaf_type
