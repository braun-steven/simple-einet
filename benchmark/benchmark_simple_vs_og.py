#!/usr/bin/env python3

from typing import List
import pickle
from itertools import product
import torch.utils.benchmark as benchmark
from torch.utils.benchmark.utils.common import TaskSpec
from simple_einet.layers.distributions import Binomial
from simple_einet.einet import Einet, EinetConfig
import os
import sys

einsum_networks_repo = os.path.join("/tmp", "EinsumNetworks")
# einsum_networks_repo = os.path.join("/home/tak/projects/", "EinsumNetworks")
if not os.path.exists(einsum_networks_repo):
    os.system(f"git clone https://github.com/cambridge-mlg/EinsumNetworks {einsum_networks_repo}")

sys.path.append(os.path.join(einsum_networks_repo, "src"))

import os
import numpy as np
import torch
from EinsumNetwork import Graph, EinsumNetwork
import argparse


def count_params(model: torch.nn.Module) -> int:
    """
    Count the number of parameters in a model.

    Args:
        model (torch.nn.Module): PyTorch model.

    Returns:
        int: Number of learnable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_og_einet(
    num_var: int,
    depth: int,
    num_sums: int,
    num_leaves: int,
    num_repetitions: int,
    num_classes: int,
    num_channels: int,
):

    graph = Graph.random_binary_trees(num_var=num_var, depth=depth, num_repetitions=num_repetitions)

    args = EinsumNetwork.Args(
        num_var=num_var,
        num_dims=num_channels,
        num_classes=num_classes,
        num_sums=num_sums,
        num_input_distributions=num_leaves,
        exponential_family=EinsumNetwork.BinomialArray,
        exponential_family_args={"N": 255},
        online_em_frequency=1,
        online_em_stepsize=0.05,
        use_em=False,
    )

    einet = EinsumNetwork.EinsumNetwork(graph, args)
    einet.initialize()
    einet.to(DEVICE)
    return einet


def make_simple_einet(
    num_var: int,
    depth: int,
    num_sums: int,
    num_leaves: int,
    num_repetitions: int,
    num_classes: int,
    num_channels: int,
):
    config = EinetConfig(
        num_features=num_var,
        num_channels=num_channels,
        depth=depth,
        num_sums=num_sums,
        num_leaves=num_leaves,
        num_repetitions=num_repetitions,
        num_classes=num_classes,
        leaf_type=Binomial,
        leaf_kwargs={"total_count": 255},
        dropout=0.0,
    )
    model = Einet(config).to(DEVICE)
    return model


def list_params(model):
    print(type(model).__name__)
    for name, module in model.named_modules():
        print(name, sum(param.numel() for param in module.parameters() if param.requires_grad))


# list_params(einet_og)
# list_params(einet_si)


def do_forward(model, x):
    model(x)


def do_backward(model, x):
    model(x).sum().backward()


def run(
    label,
    sub_label,
    batch_size,
    num_features,
    depth,
    num_sums,
    num_leaves,
    num_repetitions,
    num_classes,
    num_channels,
    results,
):

    einet_og = make_og_einet(
        num_features, depth, num_sums, num_leaves, num_repetitions, num_classes, num_channels
    )
    einet_si = make_simple_einet(
        num_features, depth, num_sums, num_leaves, num_repetitions, num_classes, num_channels
    )
    assert count_params(einet_og) == count_params(einet_si)

    x_si = torch.randint(low=0, high=256, size=(batch_size, num_channels, num_features)).to(DEVICE)
    x_og = x_si.clone().permute(0, 2, 1)  # OG impl uses [B, F, C]

    for (name, method) in [("forward", do_forward), ("backward", do_backward)]:
        results.append(
            benchmark.Timer(
                stmt="f(model, x)",
                globals={"x": x_si, "model": einet_si, "f": method},
                label=label + "-" + name,
                sub_label=sub_label,
                description="simple-einet",
            ).blocked_autorange(min_run_time=1)
        )
        results.append(
            benchmark.Timer(
                stmt="f(model, x)",
                globals={"x": x_og, "model": einet_og, "f": method},
                label=label + "-" + name,
                sub_label=sub_label,
                description="EinsumNetworks",
            ).blocked_autorange(min_run_time=1)
        )


def power_2_range(start=0, end=0):
    for i in range(start, end):
        yield 2 ** i


def main():
    # Define hyper-param ranges
    hparams = {
        "batch_size": power_2_range(end=12),
        "num_features": power_2_range(start=2, end=13),
        "depth": range(1, 10),
        "num_sums": power_2_range(end=8),
        "num_leaves": power_2_range(end=8),
        "num_channels": power_2_range(end=8),
        "num_repetitions": power_2_range(end=8),
        "num_classes": power_2_range(end=8),
    }

    # Default values
    batch_size = 256
    num_features = 512
    depth = 5
    num_sums = 32
    num_leaves = 32
    num_repetitions = 32
    num_channels = 1
    num_classes = 1

    # Compare takes a list of measurements which we'll save in results.
    results = []
    for key, values in hparams.items():
        for v in values:
            # Set kwargs using the default hyperparams
            kwargs = {
                "batch_size": batch_size,
                "num_features": num_features,
                "depth": depth,
                "num_sums": num_sums,
                "num_leaves": num_leaves,
                "num_channels": num_channels,
                "num_repetitions": num_repetitions,
                "num_classes": num_classes,
            }

            # Overwrite the hyper-parameter that is benchmarked in this loop
            kwargs[key] = v

            # Adjust depth if number of features is smaller than default 2**depth
            max_depth = int(np.log2(kwargs["num_features"]))
            kwargs["depth"] = min(kwargs["depth"], max_depth)

            print(f"{key}: {v}")

            # Run the benchmark for this hyper-parameter setting
            try:
                run(label=key, results=results, sub_label=f"{v:>4d}", **kwargs)
            except Exception as e:
                print("Failed... out of memory")

    with open("results.pkl", "wb") as f:
        f.write(pickle.dumps(results))

    compare = benchmark.Compare(results)
    compare.colorize()
    compare.print()


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser("Einsum networks benchmark.")
    parser.add_argument("--gpu", type=int, default=0, help="Device on which to run the benchmark.")
    parser.add_argument("--print-only", action="store_true", help="Only print results")
    ARGS = parser.parse_args()

    # Set global device
    DEVICE = f"cuda:{ARGS.gpu}" if torch.cuda.is_available() else "cpu"

    if ARGS.print_only:
        with open("results.pkl", "rb") as f:
            results = pickle.load(f)
        compare = benchmark.Compare(results)
        # compare.colorize()
        compare.print()

    else:
        main()
