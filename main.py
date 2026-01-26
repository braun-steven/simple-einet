#!/usr/bin/env python3
# install(suppress=[torch])
import math
import os
from typing import Union
from lion_pytorch import Lion

import numpy as np
import tqdm

from args import parse_args
from simple_einet.data import build_dataloader, get_data_shape, get_data_num_classes
from simple_einet.dist import DataType, Dist, get_data_type_from_dist, Domain
from simple_einet.layers.distributions.categorical import Categorical
from simple_einet.layers.distributions.piecewise_linear import PiecewiseLinear
from simple_einet.utils import preprocess


import torch
from torch.nn import functional as F
import torchvision

from simple_einet.layers.distributions.binomial import Binomial
from simple_einet.layers.distributions.normal import RatNormal, Normal
from simple_einet.einet import Einet, EinetConfig, posterior
from simple_einet.mixture import Mixture

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


def train_for_iterations(
    args,
    model: Union[Einet, Mixture],
    device,
    train_loader,
    optimizer,
    iterations: int,
    lr_scheduler=None,
    validation_steps: set[int] | None = None,
    val_loader=None,
):
    model.train()

    train_iter = iter(train_loader)
    pbar = tqdm.trange(1, iterations + 1)
    for step in pbar:
        # Stop after a few batches in debug mode
        if args.debug and step > 3:
            break

        data, target = next(train_iter)

        # Prepare data
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

        if args.classification:
            # Convert log p(x | y) to log p(y | x) for NLL loss.
            log_post = posterior(outputs, num_classes)
            loss = F.nll_loss(log_post, target, reduction="mean")
        else:
            loss = log_likelihoods(outputs).mean()
            loss = -1 * loss

        # Compute gradients
        fabric.backward(loss)

        # Update weights
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        # Logging
        if step % args.log_interval == 0:
            if args.classification:
                log_post = posterior(outputs, num_classes)
                _, predicted = log_post.max(1)
                correct = predicted.eq(target).sum().item()
                acc_term = " Accuracy: {:.2f}".format(100.0 * correct / len(data))
            else:
                acc_term = ""
            pbar.set_description(
                f"Train Iter: {step}/{iterations} Loss: {loss.item():.2f}{acc_term}"
            )
            if args.dry_run:
                break

        if (
            validation_steps is not None
            and val_loader is not None
            and step in validation_steps
            and not args.dry_run
        ):
            test(model, device, val_loader, f"Val ({step}/{iterations})")
            model.train()


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

            if args.classification:
                # Convert log p(x | y) to log p(y | x) for NLL loss.
                log_post = posterior(outputs, num_classes)
                test_loss += F.nll_loss(log_post, target, reduction="sum")
                _, predicted = log_post.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
            else:
                lls = log_likelihoods(outputs)
                test_loss += -1 * lls.sum()
                test_losses += lls.squeeze().cpu().tolist()

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

    if args.device == "cuda" and args.gpu is not None:
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(device)
    else:
        device = torch.device(args.device)
    # digits = [0, 1, 5, 8]
    # digits = list(range(10))
    # digits = [0, 1]
    num_classes = get_data_num_classes(args.dataset) if args.classification else 1

    # Construct Einet
    # num_classes = len(digits) if args.classification else 1
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
    if args.image_size is not None:
        data_shape = data_shape.__class__(data_shape.channels, args.image_size, args.image_size)
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
        dropout=args.dropout,
        structure=args.structure,
    )

    fabric_devices = args.num_devices
    if args.device == "cuda" and args.gpu is not None:
        fabric_devices = [int(args.gpu)]
    fabric = L.Fabric(accelerator=args.device, devices=fabric_devices, precision="16-mixed")
    fabric.launch()
    model = Einet(config)
    print(
        "Number of parameters:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    has_gauss_dist = type(model.leaf.base_leaf) in (Normal, RatNormal)

    # Optimize Einet parameters (weights and leaf params)
    optimizer = Lion(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    model, optimizer = fabric.setup(model, optimizer)

    print(model)
    home_dir = os.getenv("HOME")
    result_dir = os.path.join(home_dir, "results", "simple-einet", args.dataset)
    os.makedirs(result_dir, exist_ok=True)

    data_dir = os.path.join("~", "data")
    num_workers = min(8, os.cpu_count())
    train_loader, val_loader, test_loader = build_dataloader(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        data_dir=data_dir,
        num_workers=num_workers,
        normalize=has_gauss_dist,
        loop=True,
        seed=args.seed,
        image_size=args.image_size,
    )

    # Build a non-looping training loader for evaluation; the looping one never exhausts.
    from exp_utils import worker_init_reset_seed

    train_eval_loader = torch.utils.data.DataLoader(
        train_loader.dataset,
        shuffle=False,
        sampler=None,
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=worker_init_reset_seed,
    )

    steps_per_epoch = math.ceil(len(train_loader.dataset) / args.batch_size)
    train_iterations = args.iterations
    if train_iterations is None:
        train_iterations = args.epochs * steps_per_epoch

    # LR schedule: drop at 66% and 90% of total training iterations.
    m1 = max(1, int(round(train_iterations * 0.66)))
    m2 = max(1, int(round(train_iterations * 0.90)))
    milestones = sorted({m1, m2})
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    validations = args.validations
    if validations is None:
        # Preserve the old default behavior of "validate once per epoch" as closely as possible.
        validations = max(1, round(train_iterations / steps_per_epoch))

    if validations <= 0:
        validation_steps: set[int] = set()
    else:
        # Linearly spaced checkpoints in (0, train_iterations], inclusive of the end.
        validation_steps = set(np.linspace(0, train_iterations, validations + 1, dtype=int).tolist()[1:])

    train_loader, train_eval_loader, val_loader, test_loader = fabric.setup_dataloaders(
        train_loader, train_eval_loader, val_loader, test_loader
    )

    if args.train:
        train_for_iterations(
            args,
            model,
            device,
            train_loader,
            optimizer,
            lr_scheduler=lr_scheduler,
            iterations=train_iterations,
            validation_steps=validation_steps,
            val_loader=val_loader,
        )

        torch.save(model.state_dict(), os.path.join(result_dir, "model.pth"))
        test(model, device, train_eval_loader, "Train")
        test(model, device, val_loader, "Val")
        test(model, device, test_loader, "Test")

    else:
        model.load_state_dict(torch.load(os.path.join(result_dir, "model.pth")))

        test(model, device, train_eval_loader, "Train")
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
