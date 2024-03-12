import itertools
import csv
import subprocess
import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from sklearn import datasets
from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset
from torch.utils.data.sampler import Sampler
from torchvision.datasets import (
    CIFAR10,
    MNIST,
    SVHN,
    CelebA,
    FakeData,
    FashionMNIST,
    LSUN,
    Flowers102,
    ImageFolder,
    LFWPeople,
)

from simple_einet.layers.distributions.binomial import Binomial
from simple_einet.layers.distributions.bernoulli import Bernoulli
from simple_einet.layers.distributions.categorical import Categorical
from simple_einet.layers.distributions.multivariate_normal import MultivariateNormal
from simple_einet.layers.distributions.normal import Normal, RatNormal


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
    if "synth" in dataset_name:
        return Shape(1, 2, 1)

    if "debd" in dataset_name:
        return Shape(
            *{
                "accidents": (1, 111, 1),
                "ad": (1, 1556, 1),
                "baudio": (1, 100, 1),
                "bbc": (1, 1058, 1),
                "bnetflix": (1, 100, 1),
                "book": (1, 500, 1),
                "c20ng": (1, 910, 1),
                "cr52": (1, 889, 1),
                "cwebkb": (1, 839, 1),
                "dna": (1, 180, 1),
                "jester": (1, 100, 1),
                "kdd": (1, 64, 1),
                "kosarek": (1, 190, 1),
                "moviereview": (1, 1001, 1),
                "msnbc": (1, 17, 1),
                "msweb": (1, 294, 1),
                "nltcs": (1, 16, 1),
                "plants": (1, 69, 1),
                "pumsb_star": (1, 163, 1),
                "tmovie": (1, 500, 1),
                "tretail": (1, 135, 1),
                "voting": (1, 1359, 1),
            }[dataset_name.replace("debd-", "")]
        )

    return Shape(
        *{
            "mnist": (1, 32, 32),
            "mnist-28": (1, 28, 28),
            "fmnist": (1, 32, 32),
            "fmnist-28": (1, 28, 28),
            "cifar": (3, 32, 32),
            "svhn": (3, 32, 32),
            "svhn-extra": (3, 32, 32),
            "celeba": (3, 64, 64),
            "celeba-small": (3, 64, 64),
            "celeba-tiny": (3, 32, 32),
            "lsun": (3, 32, 32),
            "fake": (3, 32, 32),
            "flowers": (3, 32, 32),
            "tiny-imagenet": (3, 32, 32),
            "lfw": (3, 32, 32),
        }[dataset_name]
    )


@torch.no_grad()
def generate_data(dataset_name: str, n_samples: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
    tag = dataset_name.replace("synth-", "")
    if tag == "2-clusters":
        centers = [[0.0, 0.0], [0.5, 0.5]]
        cluster_stds = 0.1
        data, y = datasets.make_blobs(
            n_samples=n_samples,
            n_features=2,
            centers=centers,
            cluster_std=cluster_stds,
            random_state=0,
        )

    elif tag == "3-clusters":
        centers = [[0.0, 0.0], [0.5, 0.5], [0.5, 0.0]]
        cluster_stds = 0.05
        data, y = datasets.make_blobs(
            n_samples=n_samples,
            n_features=2,
            centers=centers,
            cluster_std=cluster_stds,
            random_state=0,
        )
    elif tag == "9-clusters":
        centers = [
            [0.0, 0.0],
            [0.5, 0.5],
            [0.5, 0.0],
            [0.0, 0.5],
            [0.5, 1.0],
            [1.0, 0.5],
            [1.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ]
        cluster_stds = 0.1
        data, y = datasets.make_blobs(
            n_samples=n_samples,
            n_features=2,
            centers=centers,
            cluster_std=cluster_stds,
            random_state=0,
        )
    elif tag == "2-moons":
        data, y = datasets.make_moons(n_samples=n_samples, noise=0.1, random_state=0)

    elif tag == "circles":
        data, y = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)

    elif tag == "aniso":
        # Anisotropicly distributed data
        X, y = datasets.make_blobs(
            n_samples=n_samples,
            cluster_std=0.2,
            random_state=0,
            centers=[[-1, -1], [-1, 0.5], [0.5, 0.5]],
        )
        transformation = [[0.5, -0.2], [-0.2, 0.4]]
        X_aniso = np.dot(X, transformation)
        data = X_aniso

    elif tag == "varied":
        # blobs with varied variances
        data, y = datasets.make_blobs(
            n_samples=n_samples,
            cluster_std=[0.5, 0.1, 0.3],
            random_state=0,
            center_box=[-2, 2],
        )
    else:
        raise ValueError(f"Invalid synthetic dataset name: {tag}.")

    data = torch.from_numpy(data).float()
    labels = torch.from_numpy(y).long()
    return data, labels


def maybe_download_debd(data_dir: str):
    if os.path.isdir(f"{data_dir}/debd"):
        return
    subprocess.run(f"git clone https://github.com/arranger1044/DEBD {data_dir}/debd".split())
    wd = os.getcwd()
    os.chdir(f"{data_dir}/debd")
    subprocess.run("git checkout 80a4906dcf3b3463370f904efa42c21e8295e85c".split())
    subprocess.run("rm -rf .git".split())
    os.chdir(wd)


def load_debd(name, data_dir, dtype="float"):
    """Load one of the twenty binary density esimtation benchmark datasets."""

    maybe_download_debd(data_dir)

    train_path = os.path.join(data_dir, "debd", "datasets", name, name + ".train.data")
    test_path = os.path.join(data_dir, "debd", "datasets", name, name + ".test.data")
    valid_path = os.path.join(data_dir, "debd", "datasets", name, name + ".valid.data")

    reader = csv.reader(open(train_path, "r"), delimiter=",")
    train_x = np.array(list(reader)).astype(dtype)

    reader = csv.reader(open(test_path, "r"), delimiter=",")
    test_x = np.array(list(reader)).astype(dtype)

    reader = csv.reader(open(valid_path, "r"), delimiter=",")
    valid_x = np.array(list(reader)).astype(dtype)

    return train_x, test_x, valid_x


def get_datasets(dataset_name, data_dir, normalize: bool) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Get the specified dataset.

    Args:
      cfg: Args.
      normalize: Normalize the dataset.

    Returns:
        Dataset: Dataset.
    """

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

    kwargs = dict(root=data_dir, download=True, transform=transform)

    # Custom split generator with fixed seed
    split_generator = torch.Generator().manual_seed(1)

    # Select the datasets
    if "synth" in dataset_name:
        # Train
        data, labels = generate_data(dataset_name, n_samples=3000)
        dataset_train = torch.utils.data.TensorDataset(data, labels)

        # Val
        data, labels = generate_data(dataset_name, n_samples=1000)
        dataset_val = torch.utils.data.TensorDataset(data, labels)

        # Test
        data, labels = generate_data(dataset_name, n_samples=1000)
        dataset_test = torch.utils.data.TensorDataset(data, labels)

    elif "debd" in dataset_name:
        # Call load_debd
        train_x, test_x, valid_x = load_debd(dataset_name.replace("debd-", ""), data_dir)
        dataset_train = torch.utils.data.TensorDataset(torch.from_numpy(train_x), torch.zeros(train_x.shape[0]))
        dataset_val = torch.utils.data.TensorDataset(torch.from_numpy(valid_x), torch.zeros(valid_x.shape[0]))
        dataset_test = torch.utils.data.TensorDataset(torch.from_numpy(test_x), torch.zeros(test_x.shape[0]))

    elif dataset_name == "digits":
        if normalize:
            transform.transforms.append(transforms.Normalize([0.5], [0.5]))

        data, labels = datasets.load_digits(return_X_y=True)
        data, labels = torch.from_numpy(data).float(), torch.from_numpy(labels).long()
        data[data == 16] = 15
        # Normalize to [0, 1]
        data = data / 15
        dataset_train = torch.utils.data.TensorDataset(data, labels)

        N = data.shape[0]
        N_train = round(N * 0.7)
        N_val = round(N * 0.2)
        N_test = N - N_train - N_val
        lenghts = [N_train, N_val, N_test]

        dataset_train, dataset_val, dataset_test = random_split(
            dataset_train, lengths=lenghts, generator=split_generator
        )

    elif dataset_name == "mnist" or dataset_name == "mnist-28":
        if normalize:
            transform.transforms.append(transforms.Normalize([0.5], [0.5]))

        dataset_train = MNIST(**kwargs, train=True)

        dataset_test = MNIST(**kwargs, train=False)

        # for dataset in [dataset_train, dataset_test]:
        #     digits = [0, 1]
        #     mask = torch.zeros_like(dataset.targets).bool()
        #     for digit in digits:
        #         mask = mask | (dataset.targets == digit)
        #
        #     dataset.data = dataset.data[mask]
        #     dataset.targets = dataset.targets[mask]

        N = len(dataset_train.data)
        N_train = round(N * 0.9)
        N_val = N - N_train
        lenghts = [N_train, N_val]

        dataset_train, dataset_val = random_split(dataset_train, lengths=lenghts, generator=split_generator)

    elif dataset_name == "fmnist" or dataset_name == "fmnist-28":
        if normalize:
            transform.transforms.append(transforms.Normalize([0.5], [0.5]))

        dataset_train = FashionMNIST(**kwargs, train=True)

        dataset_test = FashionMNIST(**kwargs, train=False)

        N = len(dataset_train.data)
        N_train = round(N * 0.9)
        N_val = N - N_train
        lenghts = [N_train, N_val]

        dataset_train, dataset_val = random_split(dataset_train, lengths=lenghts, generator=split_generator)

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

        dataset_train, dataset_val = random_split(dataset_train, lengths=lenghts, generator=split_generator)
        dataset_test = CIFAR10(**kwargs, train=False)

    elif "svhn" in dataset_name:
        if normalize:
            transform.transforms.append(transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))

        # Load train
        dataset_train = SVHN(**kwargs, split="train")

        N = len(dataset_train.data)
        lenghts = [round(N * 0.9), round(N * 0.1)]

        dataset_train, dataset_val = random_split(dataset_train, lengths=lenghts, generator=split_generator)
        dataset_test = SVHN(**kwargs, split="test")

        if dataset_name == "svhn-extra":
            # Merge train and extra into train
            dataset_extra = SVHN(**kwargs, split="extra")
            dataset_train = ConcatDataset([dataset_train, dataset_extra])

    elif dataset_name == "lsun":
        if normalize:
            transform.transforms.append(transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))

        del kwargs["download"]

        kwargs["root"] = os.path.join(kwargs["root"], "lsun")

        # Load train
        dataset_train = LSUN(**kwargs, classes=["church_outdoor_train"])
        dataset_test = LSUN(**kwargs, classes=["church_outdoor_val"])

        N = dataset_train.length
        lenghts = [round(N * 0.9), round(N * 0.1)]
        dataset_train, dataset_val = random_split(dataset_train, lengths=lenghts, generator=split_generator)

    elif dataset_name == "fake":
        # Load train
        dataset_train = FakeData(size=3000, image_size=shape, num_classes=10, transform=transform)
        dataset_val = FakeData(size=3000, image_size=shape, num_classes=10, transform=transform)
        dataset_test = FakeData(size=3000, image_size=shape, num_classes=10, transform=transform)

    elif dataset_name == "flowers":
        if normalize:
            transform.transforms.append(transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))

        # Load train
        dataset_train = Flowers102(**kwargs, split="train")
        dataset_val = Flowers102(**kwargs, split="val")
        dataset_test = Flowers102(**kwargs, split="test")

    elif dataset_name == "tiny-imagenet":
        if normalize:
            transform.transforms.append(transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))

        # Load train
        dataset_train = ImageFolder(root=os.path.join(data_dir, "tiny-imagenet-200", "train"), transform=transform)
        dataset_val = ImageFolder(root=os.path.join(data_dir, "tiny-imagenet-200", "val"), transform=transform)
        dataset_test = ImageFolder(root=os.path.join(data_dir, "tiny-imagenet-200", "test"), transform=transform)

    elif dataset_name == "lfw":
        if normalize:
            transform.transforms.append(transforms.Normalize([0.5], [0.5]))

        dataset_train = LFWPeople(**kwargs, split="train")

        dataset_test = LFWPeople(**kwargs, split="test")

        N = len(dataset_train.data)
        N_train = round(N * 0.9)
        N_val = N - N_train
        lenghts = [N_train, N_val]

        dataset_train, dataset_val = random_split(dataset_train, lengths=lenghts, generator=split_generator)

    else:
        raise Exception(f"Unknown dataset: {dataset_name}")

    return dataset_train, dataset_val, dataset_test


def build_dataloader(
    dataset_name, data_dir, batch_size, num_workers, loop: bool, normalize: bool
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # Get dataset objects
    dataset_train, dataset_val, dataset_test = get_datasets(dataset_name, data_dir, normalize=normalize)

    # Build data loader
    loader_train = _make_loader(batch_size, num_workers, dataset_train, loop=loop, shuffle=True)
    loader_val = _make_loader(batch_size, num_workers, dataset_val, loop=loop, shuffle=False)
    loader_test = _make_loader(batch_size, num_workers, dataset_test, loop=loop, shuffle=False)
    return loader_train, loader_val, loader_test


def _make_loader(batch_size, num_workers, dataset: Dataset, loop: bool, shuffle: bool) -> DataLoader:
    if loop:
        sampler = TrainingSampler(size=len(dataset))
    else:
        sampler = None

    from exp_utils import worker_init_reset_seed

    return DataLoader(
        dataset,
        shuffle=(sampler is None) and shuffle,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
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
    MULTIVARIATE_NORMAL = "multivariate_normal"
    NORMAL_RAT = "normal_rat"
    BINOMIAL = "binomial"
    CATEGORICAL = "categorical"
    BERNOULLI = "bernoulli"


def get_distribution(dist: Dist, cfg):
    """
    Get the distribution for the leaves.

    Args:
        dist: The distribution to use.

    Returns:
        leaf_type: The type of the leaves.
        leaf_kwargs: The kwargs for the leaves.

    """
    if dist == Dist.NORMAL:
        leaf_type = Normal
        leaf_kwargs = {}
    elif dist == Dist.NORMAL_RAT:
        leaf_type = RatNormal
        leaf_kwargs = {"min_sigma": cfg.min_sigma, "max_sigma": cfg.max_sigma}
    elif dist == Dist.BINOMIAL:
        leaf_type = Binomial
        leaf_kwargs = {"total_count": 2**cfg.n_bits - 1}
    elif dist == Dist.CATEGORICAL:
        leaf_type = Categorical
        leaf_kwargs = {"num_bins": 2**cfg.n_bits - 1}
    elif dist == Dist.MULTIVARIATE_NORMAL:
        leaf_type = MultivariateNormal
        leaf_kwargs = {"cardinality": cfg.multivariate_cardinality}
    elif dist == Dist.BERNOULLI:
        leaf_type = Bernoulli
        leaf_kwargs = {}
    else:
        raise ValueError(f"Unknown distribution ({dist}).")
    return leaf_kwargs, leaf_type
