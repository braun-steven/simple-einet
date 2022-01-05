#!/usr/bin/env python3

import logging
import random
import numpy as np
import datetime
from torch.utils.data.sampler import Sampler
import math
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10, CelebA, LSUN, MNIST, SVHN
import torch
import os
from torchvision import datasets, transforms
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def get_mnist(digits: List[int], batch_size: int, test_batch_size: int, debug=False):
    train_kwargs = {"batch_size": batch_size}
    test_kwargs = {"batch_size": test_batch_size}

    transform = transforms.Compose([transforms.ToTensor()])
    data_dir = os.path.join("~", "data")
    dataset_train = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    dataset_test = datasets.MNIST(data_dir, train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset_train, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset_test, **test_kwargs)

    mask = torch.zeros_like(train_loader.dataset.targets).bool()
    for digit in digits:
        mask = mask | (train_loader.dataset.targets == digit)

    train_loader.dataset.data = train_loader.dataset.data[mask]
    train_loader.dataset.targets = train_loader.dataset.targets[mask]
    mask = torch.zeros_like(test_loader.dataset.targets).bool()
    for digit in digits:
        mask = mask | (test_loader.dataset.targets == digit)
    test_loader.dataset.data = test_loader.dataset.data[mask]
    test_loader.dataset.targets = test_loader.dataset.targets[mask]

    if debug:
        train_loader.dataset.data = train_loader.dataset.data[:500]
        train_loader.dataset.targets = train_loader.dataset.targets[:500]
        test_loader.dataset.data = test_loader.dataset.data[:500]
        test_loader.dataset.targets = test_loader.dataset.targets[:500]

    return train_loader, test_loader


def colorize_mnist(images):
    images = images.expand((3, -1, -1))
    cs = torch.rand(3, 1, 1) * 0.7 + 0.3
    images_colored = images * cs
    return images_colored


def get_data_shape(dataset_name: str) -> Tuple[int, int, int]:
    """Get the expected data shape.

    Args:
        dataset_name (str): Dataset name.

    Returns:
        Tuple[int, int, int]: Tuple of [channels, height, width].
    """
    return {
        "mnist": (1, 28, 28),
        "cmnist": (3, 28, 28),
        "cifar": (3, 32, 32),
        "svhn": (3, 32, 32),
        "lsun-bedroom": (3, 64, 64),
        "lsun-tower": (3, 64, 64),
        "lsun-church-outdoor": (3, 64, 64),
        "celeba": (3, 128, 128),
        "celeba-small": (3, 64, 64),
    }[dataset_name]


def get_dataset(dataset_name: str, data_dir: str, train: bool) -> Dataset:
    """
    Get the specified dataset.

    Args:
      train: Flag for train (true) or test (false) split.

    Returns:
        Dataset: Dataset.
    """

    # Get the image size (assumes quadratic images)
    image_size = get_data_shape(dataset_name)[-1]

    # Compose image transformations
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    kwargs = dict(root=data_dir, download=True, transform=transform)

    # Select the datasets
    if dataset_name == "mnist":
        transform.transforms.pop(2)  # Remove hflip
        dataset = MNIST(**kwargs, train=train)

        digits = [2]
        mask = torch.zeros_like(dataset.targets).bool()
        for digit in digits:
            mask = mask | (dataset.targets == digit)

        dataset.data = dataset.data[mask]
        dataset.targets = dataset.targets[mask]

    elif dataset_name == "cmnist":
        transform.transforms.pop(2)  # Remove hflip
        transform.transforms.append(transforms.Lambda(colorize_mnist))
        dataset = MNIST(**kwargs, train=train)

    elif dataset_name == "celeba":
        dataset = CelebA(**kwargs, split="train" if train else "test")

    elif dataset_name == "celeba-small":
        # Automatically resized to 64x64 in transform
        dataset = CelebA(**kwargs, split="train" if train else "test")

    elif dataset_name == "cifar":

        transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.Pad(int(math.ceil(image_size * 0.04)), padding_mode="edge"),
                transforms.RandomAffine(degrees=0, translate=(0.04, 0.04)),
                transforms.CenterCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

        kwargs = dict(root=data_dir, download=True, transform=transform)

        dataset = CIFAR10(**kwargs, train=train)

    elif dataset_name == "svhn":

        dataset = SVHN(**kwargs, split="train" if train else "test")

    elif dataset_name.startswith("lsun"):
        transform = transforms.Compose(
            [
                transforms.Resize(96),
                transforms.Pad(int(math.ceil(96 * 0.04)), padding_mode="edge"),
                transforms.RandomAffine(degrees=0, translate=(0.04, 0.04)),
                transforms.CenterCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

        kwargs = dict(root=data_dir, download=True, transform=transform)

        if dataset_name == "lsun-bedroom":
            classes = "bedroom"
        elif dataset_name == "lsun-church-outdoor":
            classes = "church_outdoor"
        elif dataset_name == "lsun-tower":
            classes = "tower"

        if train:
            classes += "_train"
        else:
            classes += "_val"

        kwargs = dict(root=data_dir, transform=transform)

        # download_and_unzip_maybe(out_dir=data_dir, set_name=classes)
        dataset = LSUN(**kwargs, classes=[classes])

    else:
        raise Exception(f"Unknown dataset: {dataset_name}")

    return dataset


def build_dataloader(
    dataset_name: str, batch_size, batch_size_test, data_dir: str
) -> Tuple[DataLoader, DataLoader]:
    # Get dataset objects
    trainset = get_dataset(dataset_name, data_dir, train=True)
    testset = get_dataset(dataset_name, data_dir, train=False)

    # Build data loader
    trainloader = _make_loader(trainset, batch_size=batch_size)
    testloader = _make_loader(testset, batch_size=batch_size_test)
    return trainloader, testloader


def _make_loader(dataset: Dataset, batch_size: int) -> DataLoader:
    return DataLoader(
        dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=worker_init_reset_seed,
    )


def worker_init_reset_seed(worker_id: int):
    """Initialize the worker by settign a seed depending on the worker id.

    Args:
        worker_id (int): Unique worker id.
    """
    initial_seed = torch.initial_seed() % 2 ** 31
    seed_all_rng(initial_seed + worker_id)


def seed_all_rng(seed=None):
    """
    Set the random seed for the RNG in torch, numpy and python.

    Args:
        seed (int): if None, will use a strong random seed.
    """
    if seed is None:
        seed = (
            os.getpid()
            + int(datetime.now().strftime("%S%f"))
            + int.from_bytes(os.urandom(2), "big")
        )
        logger.info("Using a generated random seed {}".format(seed))
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
