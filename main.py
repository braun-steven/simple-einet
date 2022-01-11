#!/usr/bin/env python3
import torch
from rich.traceback import install
import tqdm
import torch.autograd.profiler as profiler

from PIL import Image
from PIL import ImageFilter
from torch._C import memory_format

from args import parse_args
from simple_einet.data import build_dataloader, get_data_shape
from simple_einet.utils import calc_bpd, preprocess

install()
# install(suppress=[torch])
import os
from typing import Union

import numpy as np
from icecream import install

install()

import torch
import torchvision

from simple_einet.distributions import Binomial, Normal, RatNormal
from simple_einet.einet import Einet, EinetConfig, EinetMixture


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


def train(args, model: Union[Einet, EinetMixture], device, train_loader, optimizer, epoch):

    cross_entropy = torch.nn.CrossEntropyLoss(reduction="sum")
    model.train()

    pbar = tqdm.tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(pbar):

        # Stop after a few batches in debug mode
        if args.debug and batch_idx > 2:
            break

        # Prepare data
        data, target = data.to(device, memory_format=torch.channels_last), target.to(device)
        data = preprocess(
            data,
            n_bits,
            n_bins,
            dequantize=True,
            has_gauss_dist=has_gauss_dist,
        )

        optimizer.zero_grad()

        # Generate outputs
        outputs = model(data)

        # Compute loss
        if args.classification:
            # In classification, compute cross entropy
            loss = cross_entropy(outputs, target)
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

            pbar.set_description(
                "Train Epoch: {} [{}/{}] Loss: {:.2f}{}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
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
            data = preprocess(
                data,
                n_bits,
                n_bins,
                dequantize=True,
                has_gauss_dist=has_gauss_dist,
            )

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
                test_losses += (
                    torch.logsumexp(outputs + np.log(outputs.shape[1]), -1).squeeze().cpu().tolist()
                )

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
        test_bpd = calc_bpd(
            torch.tensor(test_losses), data_shape, has_gauss_dist=has_gauss_dist, n_bins=n_bins
        )
        print("{} set: Bits per dim: {:.4f}".format(tag, test_bpd))

    print()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)

    n_bits = args.n_bits
    n_bins = 2 ** n_bits

    device = torch.device(args.device)
    # digits = [0, 1, 5, 8]
    digits = list(range(10))
    # digits = [0, 1]

    # Construct Einet
    num_classes = len(digits) if args.classification else 1
    # num_classes = 18
    data_shape = get_data_shape(args.dataset)

    config = EinetConfig(
        num_features=np.prod(data_shape[1:]),
        num_channels=data_shape[0],
        depth=args.D,
        num_sums=args.S,
        num_leaves=args.I,
        num_repetitions=args.R,
        num_classes=num_classes,
        leaf_type=Binomial,
        leaf_kwargs={"total_count": n_bins - 1},
        # leaf_base_class=MultivariateNormal,
        # leaf_base_kwargs={"cardinality": 2, "min_sigma": 1e-5, "max_sigma": 0.1},
        # leaf_type=RatNormal,
        # leaf_kwargs={"min_sigma": 1e-3, "max_sigma": 2.0},
        dropout=0.0,
    )
    if args.mixture > 1:
        model = EinetMixture(n_components=args.mixture, einet_config=config).to(device)
    else:
        model = Einet(config).to(device)
    print("Number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    if type(model) == EinetMixture:
        has_gauss_dist = type(model.einets[0].leaf.base_leaf) in (Normal, RatNormal)
    else:
        has_gauss_dist = type(model.leaf.base_leaf) in (Normal, RatNormal)

    # Optimize Einet parameters (weights and leaf params)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=5e-1, verbose=True)

    print(model)
    home_dir = os.getenv("HOME")
    result_dir = os.path.join(home_dir, "results", "simple-einet", "mnist")
    os.makedirs(result_dir, exist_ok=True)

    data_dir = os.path.join("~", "data")
    train_loader, test_loader = build_dataloader(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        batch_size_test=args.batch_size,
        data_dir=data_dir,
    )

    if args.train:

        if type(model) == EinetMixture:
            init_data = preprocess(
                torch.tensor(train_loader.dataset.data).to(device),
                n_bits,
                n_bins,
                dequantize=True,
                has_gauss_dist=has_gauss_dist,
            ).to(device)
            model.initialize(init_data)

        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch)
            test(model, device, test_loader, "Test")
            lr_scheduler.step()

            # ic(model.einsum_layers[0].weights.view(-1).softmax(dim=0))
            # model.sample(1)

        torch.save(model.state_dict(), os.path.join(result_dir, "model.pth"))

    else:
        model.load_state_dict(torch.load(os.path.join(result_dir, "model.pth")))
        model = model.to(device)

        test(model, device, test_loader, "Train")
        test(model, device, test_loader, "Test")

    # Don't sample when doing classification
    if not args.classification:

        model.eval()

        #######################
        # Some random samples #
        #######################
        if type(model) == Einet:
            samples = model.sample(
                num_samples=64,
                temperature_sums=args.temperature_sums,
                temperature_leaves=args.temperature_leaves,
            )
        else:
            samples = model.sample(
                num_samples_per_cluster=8,
                temperature_sums=args.temperature_sums,
                temperature_leaves=args.temperature_leaves,
            )

        samples = samples.view(-1, *data_shape)
        if not has_gauss_dist:
            grid_kwargs = dict(nrow=8, normalize=True)
            samples = samples / 255
        else:
            grid_kwargs = dict(nrow=8, normalize=True, value_range=(-0.5, 0.5))

        grid = torchvision.utils.make_grid(samples, **grid_kwargs)
        # img = Image.fromarray(np.uint8(grid.permute(1, 2, 0).cpu().numpy() * 255))
        # img = img.filter(ImageFilter.GaussianBlur(1))
        # img = img.filter(ImageFilter.SHARPEN);
        # img = img.filter(ImageFilter.SHARPEN);
        # img = img.filter(ImageFilter.SHARPEN);
        # grid = torch.from_numpy(np.array(img, dtype=float) / 255).permute(2, 0, 1)

        torchvision.utils.save_image(grid, os.path.join(result_dir, "samples.png"))

        #######
        # MPE #
        #######
        mpe = model.mpe(evidence=None)
        mpe = mpe.view(-1, *data_shape)

        torchvision.utils.save_image(mpe, os.path.join(result_dir, "mpe.png"), **grid_kwargs)

        ################
        # ground-truth #
        ################
        test_x, _ = next(iter(test_loader))
        test_x = test_x[:64].to(device)
        test_x = preprocess(
            test_x,
            n_bits,
            n_bins,
            dequantize=False,
            has_gauss_dist=has_gauss_dist,
        ).float()

        grid = torchvision.utils.make_grid(test_x.view(-1, *data_shape), **grid_kwargs)
        torchvision.utils.save_image(grid, os.path.join(result_dir, "ground_truth.png"))

        ###################
        # reconstructions #
        ###################
        image_scope = np.array(range(np.prod(data_shape))).reshape(data_shape)
        marginalized_scopes = list(image_scope[:, 0 : round(data_shape[-1] / 2), :].reshape(-1))

        num_samples = 1
        reconstructions = None
        for k in range(num_samples):
            if reconstructions is None:
                reconstructions = model.sample(
                    evidence=test_x,
                    temperature_leaves=args.temperature_leaves,
                    marginalized_scopes=marginalized_scopes,
                ).cpu()
            else:
                reconstructions += model.sample(
                    evidence=test_x,
                    temperature_leaves=args.temperature_leaves,
                    marginalized_scopes=marginalized_scopes,
                ).cpu()
        reconstructions = reconstructions.float() / num_samples
        if not has_gauss_dist:
            reconstructions = reconstructions / 255
        reconstructions = reconstructions.squeeze()

        reconstructions = reconstructions.view(-1, *data_shape)
        grid = torchvision.utils.make_grid(reconstructions, **grid_kwargs)
        torchvision.utils.save_image(grid, os.path.join(result_dir, "reconstructions.png"))

        #######################
        # reconstructions-mpe #
        #######################
        reconstructions_mpe = model.mpe(
            evidence=test_x, marginalized_scopes=marginalized_scopes
        ).cpu()
        if not has_gauss_dist:
            reconstructions_mpe = reconstructions_mpe / 255
        reconstructions_mpe = reconstructions_mpe.squeeze()

        reconstructions_mpe = reconstructions_mpe.view(-1, *data_shape)
        grid = torchvision.utils.make_grid(reconstructions_mpe, **grid_kwargs)
        torchvision.utils.save_image(grid, os.path.join(result_dir, "reconstructions_mpe.png"))

        print(f"Result directory: {result_dir}")
        print("Done.")

