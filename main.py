#!/usr/bin/env python3
# install(suppress=[torch])
import os
from typing import Union

import numpy as np
import tqdm
from icecream import install

from args import parse_args
from simple_einet.data import build_dataloader, get_data_shape
from simple_einet.dist import DataType, Dist, get_data_type_from_dist, Domain
from simple_einet.layers.distributions.categorical import Categorical
from simple_einet.layers.distributions.piecewise_linear import PiecewiseLinear
from simple_einet.utils import preprocess

install()

import torch
from torch.nn import functional as F
import torchvision

from simple_einet.layers.distributions.binomial import Binomial
from simple_einet.layers.distributions.normal import RatNormal, Normal
from simple_einet.einet import Einet, EinetConfig
from simple_einet.einet_mixture import EinetMixture

import lightning as L


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
    model.train()

    pbar = tqdm.tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(pbar):
        # Stop after a few batches in debug mode
        if args.debug and batch_idx > 2:
            break

        # Prepare data
        data = preprocess(
            data,
            n_bits,
            n_bins,
            dequantize=True,
            has_gauss_dist=has_gauss_dist,
        )

        optimizer.zero_grad()

        if args.dist == Dist.PIECEWISE_LINEAR:
            cache_leaf = True
            cache_index = batch_idx
        else:
            cache_leaf = False
            cache_index = None

        # Generate outputs
        outputs = model(data, cache_leaf=cache_leaf, cache_index=cache_index)

        if args.classification:
            model.posterior(data)
            loss = F.nll_loss(outputs, target, reduction="mean")
        else:
            loss = log_likelihoods(outputs).mean()
            loss = -1 * loss

        # Compute gradients
        fabric.backward(loss)

        # Update weights
        optimizer.step()

        # Logging
        if batch_idx % args.log_interval == 0:
            if args.classification:
                _, predicted = outputs.max(1)
                correct = predicted.eq(target).sum().item()
                acc_term = " Accuracy: {:.2f}".format(100.0 * correct / len(data))
            else:
                acc_term = ""
            pbar.set_description(
                "Train Epoch: {} [{}/{}] Loss: {:.2f}{}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    loss.item(),
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

            if args.classification:
                _, predicted = outputs.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        if args.classification:
            print("Accuracy: {:.2f}".format(100.0 * correct / total))

    test_loss /= len(loader.dataset)

    print()
    print("{} set: Average loss: {:.4f}".format(tag, test_loss))
    print()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)

    n_bits = args.n_bits
    n_bins = 2**n_bits

    device = torch.device(args.device)
    # digits = [0, 1, 5, 8]
    digits = list(range(10))
    # digits = [0, 1]

    # Construct Einet
    num_classes = len(digits) if args.classification else 1
    if args.dist == "binomial":
        leaf_type = Binomial
        leaf_kwargs = {"total_count": n_bins - 1}
    elif args.dist == "normal":
        leaf_type = Normal
        leaf_kwargs = {}
    elif args.dist == "normal_rat":
        leaf_type = RatNormal
        leaf_kwargs = {"min_sigma": args.min_sigma, "max_sigma": args.max_sigma}
    elif args.dist == "categorical":
        leaf_type = Categorical
        leaf_kwargs = {"num_bins": n_bins}
    elif args.dist == "piecewise_linear":
        leaf_type = PiecewiseLinear
        leaf_kwargs = {}

    # num_classes = 18
    data_shape = get_data_shape(args.dataset)
    num_features = np.prod(data_shape[1:])
    config = EinetConfig(
        num_features=num_features,
        num_channels=data_shape[0],
        depth=args.D,
        num_sums=args.S,
        num_leaves=args.I,
        num_repetitions=args.R,
        num_classes=num_classes,
        leaf_type=leaf_type,
        leaf_kwargs=leaf_kwargs,
        layer_type=args.layer,
        dropout=0.0,
    )

    fabric = L.Fabric(accelerator=args.device, devices=args.num_devices, precision="16-mixed")
    fabric.launch()
    model = Einet(config)
    print(
        "Number of parameters:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    has_gauss_dist = type(model.leaf.base_leaf) in (Normal, RatNormal)

    # Optimize Einet parameters (weights and leaf params)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=1e-1, verbose=True)

    model, optimizer = fabric.setup(model, optimizer)

    print(model)
    home_dir = os.getenv("HOME")
    result_dir = os.path.join(home_dir, "results", "simple-einet", args.dataset)
    os.makedirs(result_dir, exist_ok=True)

    data_dir = os.path.join("~", "data")
    train_loader, val_loader, test_loader = build_dataloader(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        data_dir=data_dir,
        num_workers=os.cpu_count(),
        normalize=False,
        loop=False,
        seed=args.seed,
    )

    train_loader, val_loader, test_loader = fabric.setup_dataloaders(train_loader, val_loader, test_loader)

    if args.dist == Dist.PIECEWISE_LINEAR:
        # Initialize the piecewise linear function
        # Collect data
        batches = []
        count = 0
        for data, _ in train_loader:
            batches.append(data)
            count += data.shape[0]
            if count > 10000:
                break
        data_init_pwl = torch.cat(batches, dim=0)

        # Prepare data
        data_init_pwl = preprocess(
            data_init_pwl,
            n_bits,
            n_bins,
            dequantize=True,
            has_gauss_dist=has_gauss_dist,
        )

        data_init_pwl = data_init_pwl.view(data_init_pwl.shape[0], data_init_pwl.shape[1], num_features)

        domains = [Domain.discrete_range(min=0, max=255)] * num_features
        with torch.no_grad():
            model.leaf.base_leaf.initialize(data_init_pwl, domains=domains)

        # Use mixture weights obtained in leaf initialization and set these to the first linsum layer weights
        model.layers[0].logits.data[:] = model.leaf.base_leaf.mixture_weights.permute(1, 0).view(1, config.num_leaves, 1, config.num_repetitions).log()

        # Visualize a couple of pixel distributions and their piecewise linear functions
        # Select 20 random pixels
        pixels = list(range(64))[::3]
        # pixels = [36, 766, 720, 588, 759, 403, 664, 428, 25, 686, 673, 638, 44, 147, 610, 470, 540, 179, 698, 420]

        d = model.leaf.base_leaf._get_base_distribution()
        log_probs = d.log_prob(data_init_pwl)

        xs = d.xs
        ys = d.ys

        for pixel in pixels:
            # Get data subset
            # xs_pixel = xs[pixel][0][0][0].squeeze()
            # ys_pixel = ys[pixel][0][0][0].squeeze()
            xs_pixel = xs[0][0][pixel][0].squeeze().cpu()
            ys_pixel = ys[0][0][pixel][0].squeeze().cpu()

            # Plot pixel distribution with pixel value as x and logprob as y values
            import matplotlib.pyplot as plt

            plt.figure(figsize=(12, 6))
            plt.plot(xs_pixel, ys_pixel, label="PWL")

            # Plot histogram of pixel values
            plt.hist(data_init_pwl[:, :, pixel].flatten().cpu().numpy(), bins=100, density=True, alpha=0.5, label="Data")
            plt.xlabel("Pixel Value")
            plt.ylabel("Density")
            plt.legend()
            plt.savefig(os.path.join(result_dir, f"pwl-{pixel}.png"), dpi=300)
            plt.close()

    if args.train:
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch)
            # lr_scheduler.step()

        torch.save(model.state_dict(), os.path.join(result_dir, "model.pth"))
        # test(model, device, train_loader, "Train")
        # test(model, device, val_loader, "Val")
        # test(model, device, test_loader, "Test")

    else:
        model.load_state_dict(torch.load(os.path.join(result_dir, "model.pth")))

        test(model, device, train_loader, "Train")
        test(model, device, val_loader, "Val")
        test(model, device, test_loader, "Test")

    # Don't sample when doing classification
    if not args.classification:
        model.eval()

        ################
        # ground-truth #
        ################
        test_x, _ = next(iter(test_loader))
        test_x = test_x[:100]
        test_x = preprocess(
            test_x,
            n_bits,
            n_bins,
            dequantize=False,
            has_gauss_dist=has_gauss_dist,
        ).float()

        if not has_gauss_dist:
            grid_kwargs = dict(nrow=10, normalize=True, padding=1, pad_value=1.0)
        else:
            grid_kwargs = dict(
                nrow=10,
                normalize=True,
                value_range=(-0.5, 0.5),
                padding=1,
                pad_value=1.0,
            )

        grid = torchvision.utils.make_grid(test_x.view(-1, *data_shape), **grid_kwargs)
        torchvision.utils.save_image(grid, os.path.join(result_dir, "ground_truth.png"))

        #######################
        # Some random samples #
        #######################
        for diff in [False, True]:
            suffix = "-diff" if diff else ""
            for mpe_at_leaves in [False, True]:
                suffix_mpe_at_leaves = "-mpe-leaves" if mpe_at_leaves else ""
                if type(model._original_module) == Einet:
                    samples = model.sample(
                        num_samples=100,
                        temperature_sums=args.temperature_sums,
                        temperature_leaves=args.temperature_leaves,
                        is_differentiable=diff,
                        mpe_at_leaves=mpe_at_leaves,
                        seed=0,
                    )
                else:
                    samples = model.sample(
                        num_samples_per_cluster=8,
                        temperature_sums=args.temperature_sums,
                        temperature_leaves=args.temperature_leaves,
                        mpe_at_leaves=mpe_at_leaves,
                        seed=0,
                    )

                samples = samples.view(-1, *data_shape)
                if not has_gauss_dist:
                    samples = samples / n_bins

                grid = torchvision.utils.make_grid(samples, **grid_kwargs)

                torchvision.utils.save_image(
                    grid, os.path.join(result_dir, f"samples{suffix}{suffix_mpe_at_leaves}.png")
                )

                ###################
                # reconstructions #
                ###################
                image_scope = np.array(range(np.prod(list(data_shape)))).reshape(data_shape)
                marginalized_scopes = list(image_scope[:, 0 : round(data_shape[-1] / 2), :].reshape(-1))

                num_samples = 1
                reconstructions = None
                for k in range(num_samples):
                    if reconstructions is None:
                        reconstructions = model.sample(
                            evidence=test_x,
                            temperature_leaves=args.temperature_leaves,
                            marginalized_scopes=marginalized_scopes,
                            mpe_at_leaves=mpe_at_leaves,
                            is_differentiable=diff,
                            seed=0,
                        ).cpu()
                    else:
                        reconstructions += model.sample(
                            evidence=test_x,
                            temperature_leaves=args.temperature_leaves,
                            marginalized_scopes=marginalized_scopes,
                            mpe_at_leaves=mpe_at_leaves,
                            is_differentiable=diff,
                            seed=0,
                        ).cpu()
                reconstructions = reconstructions.float() / num_samples
                if not has_gauss_dist:
                    reconstructions = reconstructions / n_bins
                reconstructions = reconstructions.squeeze()

                reconstructions = reconstructions.view(-1, *data_shape)
                grid = torchvision.utils.make_grid(reconstructions, **grid_kwargs)
                torchvision.utils.save_image(
                    grid, os.path.join(result_dir, f"reconstructions{suffix}{suffix_mpe_at_leaves}.png")
                )

            #######
            # MPE #
            #######
            mpe = model.mpe(evidence=None, is_differentiable=diff)
            mpe = mpe.view(-1, *data_shape)

            torchvision.utils.save_image(mpe, os.path.join(result_dir, f"mpe{suffix}.png"), **grid_kwargs)

            #######################
            # reconstructions-mpe #
            #######################
            reconstructions_mpe = model.mpe(
                evidence=test_x, marginalized_scopes=marginalized_scopes, is_differentiable=diff
            ).cpu()
            if not has_gauss_dist:
                reconstructions_mpe = reconstructions_mpe / n_bins
            reconstructions_mpe = reconstructions_mpe.squeeze()

            reconstructions_mpe = reconstructions_mpe.view(-1, *data_shape)
            grid = torchvision.utils.make_grid(reconstructions_mpe, **grid_kwargs)
            torchvision.utils.save_image(grid, os.path.join(result_dir, f"reconstructions_mpe{suffix}.png"))

        print(f"Result directory: {result_dir}")
        print("Done.")
