#!/usr/bin/env python3

from typing import List
import numpy as np
import torch
from torch import nn
from clipper import DistributionClipper

from layers import EinsumLayer, Product, Sum
from utils import SamplingContext


class Einet(nn.Module):
    def __init__(self, K, D, R, in_features, leaf_cls):
        """EinsumNetwork Module.

        Args:
            K (int): Number of sum nodes in each layer.
            D (int): Depth.
            R (int): Number of repetitions.
            in_features (int): Number of input features.
            leaf_cls: Leaf layer class.
        """
        super().__init__()
        self.in_features = in_features
        self.in_channels = K
        self.num_sums = K
        self.depth = D
        self.num_repetitions = R
        self.rand_perm = torch.tensor(np.random.permutation(self.in_features))

        # Create leaf layer
        self.leaf = leaf_cls(in_features=in_features, out_channels=K, num_repetitions=R)

        layers = []
        _in_features = in_features
        _in_channels = self.in_channels
        for d in range(self.depth):
            # Last channel should output only a single sum node
            if d < self.depth - 1:
                out_channels = self.num_sums
            else:
                out_channels = 1

            l = EinsumLayer(
                in_features=_in_features,
                in_channels=_in_channels,
                out_channels=out_channels,
                num_repetitions=R,
            )
            layers.append(l)
            _in_features = l._out_features
            _in_channels = self.num_sums

        self.layers = nn.ModuleList(layers)
        self.prod = Product(_in_features, cardinality=_in_features)
        self.root = Sum(in_channels=R, in_features=1, out_channels=1, num_repetitions=1)

    def forward(self, x: torch.Tensor):
        log_p = self.leaf(x)

        # Forward through all layers, bottom up
        for layer in self.layers:
            log_p = layer(log_p)

        # Root layer merges all scopes that are left
        log_p = self.prod(log_p)

        # Shift repetition dimension to build sum over repetitions
        log_p = log_p.permute(0, 1, 3, 2)
        log_p = self.root(log_p)
        log_p = log_p.view(x.shape[0])

        return log_p

    def sample(self, n: int) -> torch.Tensor:
        """Sample from the Einet model.

        Args:
            n (int): Number of samples to generate.

        Returns:
            torch.Tensor: Generated samples.
        """
        context = SamplingContext(n=n)

        context = self.root.sample(context=context)

        # Exchange parent and repetition indices since thie root layer
        # models a mixture over the repetitions
        context = SamplingContext(
            n=context.n,
            parent_indices=context.repetition_indices.unsqueeze(1),
            repetition_indices=context.parent_indices.squeeze(1),
        )

        # Sample from scope merging product layer
        context = self.prod.sample(context.n, context)

        # Sample from all other (EisumLayers) layers in reverse (top-down)
        for layer in reversed(self.layers):
            context = layer.sample(context.n, context)

        # Sample from leaf layer
        samples = self.leaf.sample(context.n, context)
        return samples


if __name__ == "__main__":
    from distributions import Normal
    torch.manual_seed(0)

    # Input dimensions
    in_features = 4
    batchsize = 5

    # Create input sample
    x = torch.randn(batchsize, in_features)

    # Construct Einet
    einet = Einet(K=2, D=2, R=2, in_features=in_features, leaf_cls=Normal)

    # Compute log-likelihoods
    lls = einet(x)
    print(f"{lls=}")
    print(f"{lls.shape=}")

    # Construct samples
    samples = einet.sample(2)
    print(f"{samples=}")
    print(f"{samples.shape=}")

    # Optimize Einet parameters (weights and leaf params)
    optim = torch.optim.Adam(einet.parameters(), lr=0.001)
    clipper = DistributionClipper()

    for _ in range(1000):
        optim.zero_grad()

        # Forward pass: log-likelihoods
        lls = einet(x)

        # Backprop NLL loss
        nlls = -1 * lls.sum()
        nlls.backward()

        # Update weights
        optim.step()

        # Clip leaf distribution parameters (e.g. std > 0.0, etc.)
        clipper(einet.leaf)
