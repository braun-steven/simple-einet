import time

from simple_einet.layers.distributions.piecewise_linear import PiecewiseLinear
import imageio.v3 as imageio
import itertools
import os
import subprocess
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple
import csv

import numpy as np
import torch
import torchvision.transforms as transforms
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
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
from simple_einet.layers.distributions.categorical import Categorical
from simple_einet.layers.distributions.multivariate_normal import MultivariateNormal
from simple_einet.layers.distributions.normal import Normal, RatNormal

import logging

logger = logging.getLogger(__name__)


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
        return Shape(2, 1, 1)

    if dataset_name in DEBD:
        shape = DEBD_shapes[dataset_name]["train"]
        return Shape(channels=1, height=shape[1], width=1)

    return Shape(
        *{
            "mnist-16": (1, 16, 16),
            "mnist-32": (1, 32, 32),
            "mnist-bin": (1, 28, 28),
            "mnist": (1, 28, 28),
            "fmnist": (1, 28, 28),
            "fmnist-16": (1, 16, 16),
            "fmnist-32": (1, 32, 32),
            "cifar": (3, 32, 32),
            "svhn": (3, 32, 32),
            "svhn-extra": (3, 32, 32),
            "celeba": (3, 128, 128),
            "celeba-small": (3, 64, 64),
            "celeba-tiny": (3, 32, 32),
            "lsun": (3, 32, 32),
            "fake": (3, 32, 32),
            "flowers": (3, 32, 32),
            "tiny-imagenet": (3, 32, 32),
            "lfw": (3, 32, 32),
            "20newsgroup": (1, 50, 1),
            "kddcup99": (1, 118, 1),
            "covtype": (1, 54, 1),
            "breast_cancer": (1, 30, 1),
            "wine": (1, 13, 1),
            "digits": (1, 8, 8),
        }[dataset_name]
    )


def get_data_num_classes(dataset_name: str) -> int:
    """Get the number of classes for a specific dataset.

    Args:
        dataset_name (str): Dataset name.

    Returns:
        int: Number of classes.
    """
    if "synth" in dataset_name:
        return 2

    if dataset_name in DEBD:
        return 0

    return {
        "mnist-16": 10,
        "mnist-32": 10,
        "mnist-bin": 10,
        "mnist": 10,
        "fmnist": 10,
        "fmnist-16": 10,
        "fmnist-32": 10,
        "cifar": 10,
        "svhn": 10,
        "svhn-extra": 10,
        "celeba": 0,
        "celeba-small": 0,
        "celeba-tiny": 0,
        "lsun": 0,
        "fake": 10,
        "flowers": 102,
        "tiny-imagenet": 200,
        "lfw": 0,
        "20newsgroup": 20,
        "kddcup99": 23,
        "covtype": 7,
        "breast_cancer": 2,
        "wine": 3,
    }[dataset_name]


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


def to_255_int(x):
    return (x * 255).int()


def maybe_download_debd(data_dir: str):
    debd_dir = os.path.join(data_dir, "debd")
    if os.path.isdir(debd_dir):
        return
    subprocess.run(["git", "clone", "https://github.com/arranger1044/DEBD", debd_dir])
    wd = os.getcwd()
    os.chdir(debd_dir)
    subprocess.run(["git", "checkout", "80a4906dcf3b3463370f904efa42c21e8295e85c"])
    subprocess.run(["rm", "-rf", ".git"])
    os.chdir(wd)


def load_debd(name, data_dir, dtype="int32"):
    """Load one of the twenty binary density esimtation benchmark datasets."""

    maybe_download_debd(data_dir)

    debd_dir = os.path.join(data_dir, "debd")

    train_path = os.path.join(debd_dir, "datasets", name, name + ".train.data")
    test_path = os.path.join(debd_dir, "datasets", name, name + ".test.data")
    valid_path = os.path.join(debd_dir, "datasets", name, name + ".valid.data")

    reader = csv.reader(open(train_path, "r"), delimiter=",")
    train_x = np.array(list(reader)).astype(dtype)

    reader = csv.reader(open(test_path, "r"), delimiter=",")
    test_x = np.array(list(reader)).astype(dtype)

    reader = csv.reader(open(valid_path, "r"), delimiter=",")
    valid_x = np.array(list(reader)).astype(dtype)

    return train_x, test_x, valid_x


DEBD = [
    "accidents",
    "ad",
    "baudio",
    "bbc",
    "bnetflix",
    "book",
    "c20ng",
    "cr52",
    "cwebkb",
    "dna",
    "jester",
    "kdd",
    "kosarek",
    "moviereview",
    "msnbc",
    "msweb",
    "nltcs",
    "plants",
    "pumsb_star",
    "tmovie",
    "tretail",
    "voting",
]

DEBD_shapes = {
    "accidents": dict(train=(12758, 111), valid=(2551, 111), test=(1700, 111)),
    "ad": dict(train=(2461, 1556), valid=(491, 1556), test=(327, 1556)),
    "baudio": dict(train=(15000, 100), valid=(3000, 100), test=(2000, 100)),
    "bbc": dict(train=(1670, 1058), valid=(330, 1058), test=(225, 1058)),
    "bnetflix": dict(train=(15000, 100), valid=(3000, 100), test=(2000, 100)),
    "book": dict(train=(8700, 500), valid=(1739, 500), test=(1159, 500)),
    "c20ng": dict(train=(11293, 910), valid=(3764, 910), test=(3764, 910)),
    "cr52": dict(train=(6532, 889), valid=(1540, 889), test=(1028, 889)),
    "cwebkb": dict(train=(2803, 839), valid=(838, 839), test=(558, 839)),
    "dna": dict(train=(1600, 180), valid=(1186, 180), test=(400, 180)),
    "jester": dict(train=(9000, 100), valid=(4116, 100), test=(1000, 100)),
    "kdd": dict(train=(180092, 64), valid=(34955, 64), test=(19907, 64)),
    "kosarek": dict(train=(33375, 190), valid=(6675, 190), test=(4450, 190)),
    "moviereview": dict(train=(1600, 1001), valid=(250, 1001), test=(150, 1001)),
    "msnbc": dict(train=(291326, 17), valid=(58265, 17), test=(38843, 17)),
    "msweb": dict(train=(29441, 294), valid=(5000, 294), test=(3270, 294)),
    "nltcs": dict(train=(16181, 16), valid=(3236, 16), test=(2157, 16)),
    "plants": dict(train=(17412, 69), valid=(3482, 69), test=(2321, 69)),
    "pumsb_star": dict(train=(12262, 163), valid=(2452, 163), test=(1635, 163)),
    "tmovie": dict(train=(4524, 500), valid=(591, 500), test=(1002, 500)),
    "tretail": dict(train=(22041, 135), valid=(4408, 135), test=(2938, 135)),
    "voting": dict(train=(1214, 1359), valid=(350, 1359), test=(200, 1359)),
}

DEBD_display_name = {
    "accidents": "accidents",
    "ad": "ad",
    "baudio": "audio",
    "bbc": "bbc",
    "bnetflix": "netflix",
    "book": "book",
    "c20ng": "20ng",
    "cr52": "reuters-52",
    "cwebkb": "web-kb",
    "dna": "dna",
    "jester": "jester",
    "kdd": "kdd-2k",
    "kosarek": "kosarek",
    "moviereview": "moviereview",
    "msnbc": "msnbc",
    "msweb": "msweb",
    "nltcs": "nltcs",
    "plants": "plants",
    "pumsb_star": "pumsb-star",
    "tmovie": "each-movie",
    "tretail": "retail",
    "voting": "voting",
}


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

    # if not normalize:
    #     transform.transforms.append(transforms.Lambda(to_255_int))

    kwargs = dict(root=data_dir, download=True, transform=transform)

    # Custom split generator with fixed seed
    split_generator = torch.Generator().manual_seed(1)

    # Select the datasets
    if "synth" in dataset_name:
        # Train
        X, labels = generate_data(dataset_name, n_samples=3000)
        dataset_train = torch.utils.data.TensorDataset(X, labels)

        # Val
        X, labels = generate_data(dataset_name, n_samples=1000)
        dataset_val = torch.utils.data.TensorDataset(X, labels)

        # Test
        X, labels = generate_data(dataset_name, n_samples=1000)
        dataset_test = torch.utils.data.TensorDataset(X, labels)

    elif dataset_name == "mnist" or dataset_name == "mnist-32" or dataset_name == "mnist-16":
        if normalize:
            transform.transforms.append(transforms.Normalize([0.5], [0.5]))

        dataset_train = MNIST(**kwargs, train=True)

        dataset_test = MNIST(**kwargs, train=False)

        # for dataset in [dataset_train, dataset_test]:
        #     import warnings
        #     warnings.warn("Using only digits 0 and 1 for MNIST.")
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

        dataset_train, dataset_val = random_split(dataset_train, lengths=lenghts, generator=split_generator)

    elif dataset_name == "mnist-bin":
        # Download binary mnist dataset
        if not os.path.exists(os.path.join(data_dir, "mnist-bin")):
            # URL of the image
            url = "https://i.imgur.com/j0SOfRW.png"
            output_filename = "mnist-bin.png"

            # Use wget to download the image
            os.system(f"curl {url} --output {output_filename}")

            # Load the downloaded image using imageio
            image = imageio.imread(output_filename)
        else:
            # Load image
            image = imageio.imread(os.path.join(data_dir, "mnist-bin.png"))

        ims, labels = np.split(image[..., :3].ravel(), [-70000])
        ims = np.unpackbits(ims).reshape((-1, 1, 28, 28))
        ims, labels = [np.split(y, [50000, 60000]) for y in (ims, labels)]

        (train_x, train_labels), (test_x, test_labels), (_, _) = (
            (ims[0], labels[0]),
            (ims[1], labels[1]),
            (ims[2], labels[2]),
        )

        # Make dataset from numpy images and labels
        dataset_train = torch.utils.data.TensorDataset(torch.tensor(train_x), torch.tensor(train_labels))
        dataset_test = torch.utils.data.TensorDataset(torch.tensor(test_x), torch.tensor(test_labels))

        # for dataset in [dataset_train, dataset_test]:
        #     import warnings
        #     warnings.warn("Using only digits 0 and 1 for MNIST.")
        #     digits = [0, 1]
        #     mask = torch.zeros_like(dataset.targets).bool()
        #     for digit in digits:
        #         mask = mask | (dataset.targets == digit)
        #
        #     dataset.data = dataset.data[mask]
        #     dataset.targets = dataset.targets[mask]

        N = len(dataset_train.tensors[0])
        N_train = round(N * 0.9)
        N_val = N - N_train
        lenghts = [N_train, N_val]

        dataset_train, dataset_val = random_split(dataset_train, lengths=lenghts, generator=split_generator)
    elif dataset_name == "digits":
        # SKlearn digits dataset
        digits = datasets.load_digits()
        X, y = digits.data, digits.target

        X = X / X.max()

        # Split into train, val, test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
        )


        # Reshape
        X_train = X_train.reshape(-1, *shape)
        X_val = X_val.reshape(-1, *shape)
        X_test = X_test.reshape(-1, *shape)

        dataset_train = torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        dataset_val = torch.utils.data.TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
        dataset_test = torch.utils.data.TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

    elif dataset_name == "fmnist" or dataset_name == "fmnist-32":
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

    elif dataset_name in DEBD:
        name = dataset_name

        # Load the DEBD dataset
        train_x, test_x, valid_x = load_debd(name, data_dir)
        shape = get_data_shape(dataset_name)
        train_x = train_x.reshape(-1, *shape)
        test_x = test_x.reshape(-1, *shape)
        valid_x = valid_x.reshape(-1, *shape)
        dataset_train = torch.utils.data.TensorDataset(torch.tensor(train_x), torch.zeros(len(train_x)))
        dataset_val = torch.utils.data.TensorDataset(torch.tensor(valid_x), torch.zeros(len(valid_x)))
        dataset_test = torch.utils.data.TensorDataset(torch.tensor(test_x), torch.zeros(len(test_x)))

    elif dataset_name == "20newsgroup":
        # Load the 20 newsgroup dataset
        from sklearn.datasets import fetch_20newsgroups_vectorized

        # Load the dataset
        X_train, y_train = fetch_20newsgroups_vectorized(return_X_y=True, data_home=data_dir, subset="train")
        X_test, y_test = fetch_20newsgroups_vectorized(return_X_y=True, data_home=data_dir, subset="test")

        # Split train into train and val
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
        )

        # Do dimensionality reduction with PCA
        pca = PCA(
            n_components=50,
        )
        logger.info("Running PCA with 50 components on 20newsgroup dataset")
        t0 = time.time()
        X_train = pca.fit_transform(X=X_train.toarray())
        duration = time.time() - t0
        logger.info(f"PCA done in {duration:.2f}s")
        X_val = pca.transform(X_val.toarray())
        X_test = pca.transform(X_test.toarray())

        # Scale with StandardScaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        X_train = X_train.reshape(-1, *shape)
        X_val = X_val.reshape(-1, *shape)
        X_test = X_test.reshape(-1, *shape)

        # Convert to float32
        X_train = X_train.astype(np.float32)
        X_val = X_val.astype(np.float32)
        X_test = X_test.astype(np.float32)

        # Construct datasets
        dataset_train = torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        dataset_val = torch.utils.data.TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
        dataset_test = torch.utils.data.TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

    elif dataset_name == "covtype":
        # Load the covtype dataset
        from sklearn.datasets import fetch_covtype

        # Load the dataset
        X, y = fetch_covtype(data_home=data_dir, return_X_y=True)
        X = X.astype(np.float32)

        # Encode Labels
        y = LabelEncoder().fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
        )

        # Apply StandardScaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        # Reshape
        X_train = X_train.reshape(-1, *shape)
        X_val = X_val.reshape(-1, *shape)
        X_test = X_test.reshape(-1, *shape)


        dataset_train = torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        dataset_val = torch.utils.data.TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
        dataset_test = torch.utils.data.TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

    elif dataset_name == "kddcup99":
        # Load the kddcup99 dataset
        from sklearn.datasets import fetch_kddcup99

        # Load the dataset
        X, y = fetch_kddcup99(data_home=data_dir, return_X_y=True)

        # Encode Labels
        y = LabelEncoder().fit_transform(y)

        # Convert the byte strings to regular strings
        X[:, 1:4] = X[:, 1:4].astype(str)

        # Identify the categorical columns (in this case, columns 1, 2, and 3)
        categorical_columns = [1, 2, 3]

        # Separate the categorical features from the numerical features
        categorical_data = X[:, categorical_columns]
        numerical_data = np.delete(X, categorical_columns, axis=1)

        # Apply OneHotEncoder to the categorical data
        encoder = OneHotEncoder(sparse=False)
        encoded_categorical_data = encoder.fit_transform(categorical_data)

        # Combine the encoded categorical features with the numerical features
        X = np.hstack((numerical_data, encoded_categorical_data))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
        )

        # Apply StandardScaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        # Reshape
        X_train = X_train.reshape(-1, *shape).astype(np.float32)
        X_val = X_val.reshape(-1, *shape).astype(np.float32)
        X_test = X_test.reshape(-1, *shape).astype(np.float32)


        # Construct datasets
        dataset_train = torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        dataset_val = torch.utils.data.TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
        dataset_test = torch.utils.data.TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

    elif dataset_name == "breast_cancer":
        # Load the breast cancer dataset
        from sklearn.datasets import load_breast_cancer

        # Load the dataset
        X, y = load_breast_cancer(return_X_y=True)
        X = X.astype(np.float32)

        # Encode Labels
        y = LabelEncoder().fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
        )

        # Apply StandardScaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        # Reshape
        X_train = X_train.reshape(-1, *shape)
        X_val = X_val.reshape(-1, *shape)
        X_test = X_test.reshape(-1, *shape)


        dataset_train = torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        dataset_val = torch.utils.data.TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
        dataset_test = torch.utils.data.TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

    elif dataset_name == "wine":
        # Load the wine dataset
        from sklearn.datasets import load_wine

        # Load the dataset
        X, y = load_wine(return_X_y=True)
        X = X.astype(np.float32)

        # Encode Labels
        y = LabelEncoder().fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
        )

        # Apply StandardScaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        # Reshape
        X_train = X_train.reshape(-1, *shape)
        X_val = X_val.reshape(-1, *shape)
        X_test = X_test.reshape(-1, *shape)

        dataset_train = torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        dataset_val = torch.utils.data.TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
        dataset_test = torch.utils.data.TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

    else:
        raise Exception(f"Unknown dataset: {dataset_name}")


    # # Ensure, that all datasets are in float
    # for dataset in [dataset_train, dataset_val, dataset_test]:
    #     if isinstance(dataset, torch.utils.data.TensorDataset):
    #         dataset.tensors = (dataset.tensors[0].float(), dataset.tensors[1].float())
    #     elif isinstance(dataset, torch.utils.data.dataset.Subset):
    #         dataset.dataset.data = dataset.dataset.data.float()
    #     else:
    #         dataset.data = dataset.data.float()


    return dataset_train, dataset_val, dataset_test


def is_1d_data(dataset_name):
    """Check if the dataset is 1D data."""
    if dataset_name in DEBD:
        return True

    if dataset_name in ["20newsgroup", "covtype", "kddcup99", "breast_cancer", "wine"]:
        return True

    if "synth" in dataset_name:
        return True

    return False


def is_classification_data(dataset_name):
    """Check if the dataset is 1D data."""
    if dataset_name in DEBD or "celeba" in dataset_name:
        return False

    return True


def build_dataloader(
        dataset_name, data_dir, batch_size, num_workers, loop: bool, normalize: bool, seed: int
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # Get dataset objects
    dataset_train, dataset_val, dataset_test = get_datasets(dataset_name, data_dir, normalize=normalize)

    # Build data loader
    loader_train = _make_loader(batch_size, num_workers, dataset_train, loop=loop, shuffle=True, seed=seed)
    loader_val = _make_loader(batch_size, num_workers, dataset_val, loop=False, shuffle=False, seed=seed)
    loader_test = _make_loader(batch_size, num_workers, dataset_test, loop=False, shuffle=False, seed=seed)
    return loader_train, loader_val, loader_test


def _make_loader(batch_size, num_workers, dataset: Dataset, loop: bool, shuffle: bool, seed: int) -> DataLoader:
    if loop:
        sampler = TrainingSampler(size=len(dataset), seed=seed)
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


