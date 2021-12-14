#!/usr/bin/env python3
import torch
from rich.traceback import install
install(suppress=[torch])
import argparse
import os
from typing import Tuple

from torch import Tensor
import numpy as np
from icecream import ic, install

install()

import torch
from simple_einet.distributions import Binomial, MultivariateNormal, Normal, RatNormal
from simple_einet.einet import Einet, EinetConfig
from torchvision import datasets, transforms
import torchvision

parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
parser.add_argument(
    "--train",
    action="store_true",
    help="Run training",
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=64,
    metavar="N",
    help="input batch size for training (default: 64)",
)
parser.add_argument(
    "--test-batch-size",
    type=int,
    default=1000,
    metavar="N",
    help="input batch size for testing (default: 1000)",
)
parser.add_argument(
    "--n-bits",
    type=int,
    default=8,
    metavar="N",
    help="number of bits for each pixel (default: 8)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=14,
    metavar="N",
    help="number of epochs to train (default: 14)",
)
parser.add_argument(
    "--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)"
)
parser.add_argument(
    "--temperature-leaves",
    type=float,
    default=1.0,
    help="Temperature for leave variance during sampling.",
)
parser.add_argument(
    "--temperature-sums",
    type=float,
    default=1.0,
    help="Temperature for sum weights during sampling.",
)
parser.add_argument(
    "--dry-run", action="store_true", default=False, help="quickly check a single pass"
)
parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
parser.add_argument(
    "--log-interval",
    type=int,
    default=10,
    metavar="N",
    help="how many batches to wait before logging training status",
)
parser.add_argument(
    "--save-model",
    action="store_true",
    default=False,
    help="For Saving the current Model",
)

parser.add_argument(
    "--classification",
    action="store_true",
    default=False,
    help="Flag for learning a discriminative task of classifying MNIST digits.",
)

parser.add_argument(
    "--device",
    default="cuda",
    help="Device flag. Can be either 'cpu' or 'cuda'.",
)
parser.add_argument("-K", type=int, default=10)
parser.add_argument("-D", type=int, default=3)
parser.add_argument("-R", type=int, default=1)
args = parser.parse_args()

torch.manual_seed(args.seed)


n_bits = args.n_bits
n_bins = 2 ** n_bits

device = torch.device(args.device)
# digits = [0, 1, 5, 8]
# digits = list(range(10))
digits = [0, 1]

# Construct Einet
num_classes = len(digits) if args.classification else 1
# num_classes = 10
config = EinetConfig(
    in_features=28 ** 2,
    D=args.D,
    S=args.K,
    I=args.K,
    R=args.R,
    C=num_classes,
    leaf_base_class=Binomial,
    leaf_base_kwargs={"total_count": 255},
    # leaf_base_class=MultivariateNormal,
    # leaf_base_kwargs={"cardinality": 2, "min_sigma": 1e-5, "max_sigma": 0.1},
    # leaf_base_class=RatNormal,
    # leaf_base_kwargs={"min_sigma": 1e-5, "max_sigma": 1.0},
    dropout=0.0,
)
model = Einet(config).to(device)
print("Number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

has_gauss_dist = type(model._leaf.base_leaf) in (Normal, RatNormal)

# Optimize Einet parameters (weights and leaf params)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=1e-1, verbose=True)

train_kwargs = {"batch_size": args.batch_size}
test_kwargs = {"batch_size": args.test_batch_size}

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


def calc_bpd(log_p: Tensor, image_shape: Tuple[int, int, int]) -> float:
    n_pixels = np.prod(image_shape)

    if has_gauss_dist:
        # https://github.com/openai/glow/blob/eaff2177693a5d84a1cf8ae19e8e0441715b82f8/model.py#L172
        bpd = log_p - np.log(n_bins) * n_pixels
        bpd = (-bpd / (np.log(2) * n_pixels)).mean()

    else:
        bpd = log_p - np.log(n_bins) * n_pixels
        bpd = (-bpd / (np.log(2) * n_pixels)).mean()

    return bpd


def pertubate_image(image: Tensor, n_bins: int) -> Tensor:
    return image + torch.rand_like(image) / n_bins


def dequantize_image(image: Tensor, n_bins: int, n_bits: int) -> Tensor:
    image = image * 255
    if n_bits < 8:
        image = torch.floor(image / 2 ** (8 - n_bits))

    image = image / n_bins - 0.5
    return image


def preprocess(
    image: Tensor, n_bits: int, image_shape: Tuple[int, int, int], device, pertubate=True
) -> Tensor:
    channels, height, width = image_shape
    image = image.to(device)
    image = image.view(-1, channels, height, width)
    image = dequantize_image(image, n_bins, n_bits)
    if pertubate:
        image = pertubate_image(image, n_bins)
    image = image.view(image.shape[0], -1)
    return image


def log_likelihoods(outputs, targets=None):
    """Compute the likelihood of an Einet."""
    if targets is None:
        num_roots = outputs.shape[-1]
        if num_roots == 1:
            lls = outputs
        else:
            num_roots = torch.tensor(float(num_roots), device=outputs.device)
            lls = torch.logsumexp(outputs - torch.log(num_roots), -1)
    else:
        lls = outputs.gather(-1, targets.unsqueeze(-1))
    return lls

def train(args, model: Einet, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # Prepare data
        data, target = data.to(device), target.to(device)
        data = data.view(-1, 28 ** 2)
        if has_gauss_dist:
            data = preprocess(data, n_bits, image_shape=(1, 28, 28), device=device, pertubate=True)
        else:
            data = (data * 255).long()
        optimizer.zero_grad()

        # Generate outputs
        outputs = model(data)

        # Compute loss
        if args.classification:
            # In classification, compute cross entropy
            lls_target = torch.gather(outputs, dim=1, index=target.unsqueeze(-1))
            norm = torch.logsumexp(outputs, -1)
            loss = -1 / data.shape[0] * (lls_target - norm).sum()
        else:
            loss = -1 * log_likelihoods(outputs).sum()

        # Compute gradients
        loss.backward()

        # Update weights
        optimizer.step()

        # Logging
        if batch_idx % args.log_interval == 0:

            if args.classification:
                # Log accuracy
                predictions = outputs.argmax(-1)
                correct = (predictions == target).sum()
                total = target.shape[0]
                acc = correct / total
                acc_term = f"\tAcc: {acc:.6f}"
            else:
                acc_term = ""

            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}{}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item() / args.batch_size,
                    acc_term,
                )
            )
            if args.dry_run:
                break


def test(model, device, loader, tag):
    model.eval()
    test_loss = 0
    test_losses = []

    if args.classification:
        correct = 0
        total = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            data = data.view(-1, 28 ** 2)
            if has_gauss_dist:
                data = preprocess(data, n_bits, image_shape=(1, 28, 28), device=device, pertubate=True)
            else:
                data = (data * 255).long()
            outputs = model(data)

            lls = log_likelihoods(outputs)
            test_loss += -1 * lls.sum()
            test_losses += lls.squeeze().cpu().tolist()

            # Else compute negative log likelihoods
            if outputs.shape[1] == 1:
                test_loss += -1 * outputs.sum()
                test_losses += outputs.squeeze().cpu().tolist()
            else:
                test_loss += -1 * torch.logsumexp(outputs + np.log(outputs.shape[1]), -1).sum()
                test_losses += torch.logsumexp(outputs + np.log(outputs.shape[1]), -1).squeeze().cpu().tolist()


            if args.classification:
                predictions = outputs.argmax(-1)
                batch_correct = (predictions == target).sum()
                batch_size = target.shape[0]
                correct += batch_correct
                total += batch_size

    test_loss /= len(loader.dataset)

    print()
    print("{} set: Average loss: {:.4f}".format(tag, test_loss))
    if args.classification:
        acc = correct / total * 100
        print("{} set: Accuracy {:.2f}".format(tag, acc))
    else:
        test_bpd = calc_bpd(torch.tensor(test_losses), (1, 28, 28))
        print("{} set: Bits per dim: {:.4f}".format(tag, test_bpd))

    print()


print(model)
home_dir = os.getenv("HOME")
result_dir = os.path.join(home_dir, "results", "simple-einet", "mnist")
os.makedirs(result_dir, exist_ok=True)

if args.train:
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        # test(model, device, train_loader, "Train")
        test(model, device, test_loader, "Test")
        lr_scheduler.step()

    torch.save(model.state_dict(), os.path.join(result_dir, "model.pth"))

else:
    model.load_state_dict(torch.load(os.path.join(result_dir, "model.pth")))
    model = model.to(device)

    test(model, device, test_loader, "Train")
    test(model, device, test_loader, "Test")

# Don't sample when doing classification
if args.classification:
    exit(0)

model.eval()

# Some random samples
samples = model.sample(
    num_samples=64,
    temperature_sums=args.temperature_sums,
    temperature_leaves=args.temperature_leaves,
)
samples = samples.view(-1, 1, 28, 28)
if not has_gauss_dist:
    grid_kwargs = dict(nrow=8, normalize=False)
    samples = samples / 255
else:
    grid_kwargs = dict(nrow=8, normalize=True, value_range=(-0.5, 0.5))

grid = torchvision.utils.make_grid(samples, **grid_kwargs)
torchvision.utils.save_image(grid, os.path.join(result_dir, "samples.png"))


image_scope = np.array(range(28 ** 2)).reshape(28, 28)
marginalized_scopes = list(image_scope[0 : round(28 / 2), :].reshape(-1))
keep_idx = [i for i in range(28 ** 2) if i not in marginalized_scopes]


test_x, _ = next(iter(test_loader))
test_x = test_x[:64].to(device).view(-1, 28 ** 2)
if has_gauss_dist:
    test_x = preprocess(test_x, n_bits=n_bits, image_shape=(1, 28, 28), device=device, pertubate=False)
else:
    test_x = (test_x * 255)


grid = torchvision.utils.make_grid(
    test_x.view(-1, 1, 28, 28) / 255, **grid_kwargs
)
torchvision.utils.save_image(grid, os.path.join(result_dir, "ground_truth.png"))

reconstructions = None

num_samples = 5
for k in range(num_samples):
    if reconstructions is None:
        reconstructions = model.sample(
            evidence=test_x, temperature_leaves=args.temperature_leaves, marginalized_scopes=marginalized_scopes
        ).cpu()
    else:
        reconstructions += model.sample(
            evidence=test_x, temperature_leaves=args.temperature_leaves, marginalized_scopes=marginalized_scopes
        ).cpu()
reconstructions = reconstructions.float() / num_samples
if not has_gauss_dist:
    reconstructions = reconstructions / 255
reconstructions = reconstructions.squeeze()

reconstructions = reconstructions.view(-1, 1, 28, 28)
grid = torchvision.utils.make_grid(reconstructions, **grid_kwargs)
torchvision.utils.save_image(grid, os.path.join(result_dir, "reconstructions.png"))

print(f"Result directory: {result_dir}")
print("Done.")
