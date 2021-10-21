#!/usr/bin/env python3

from typing import List
import numpy as np
import torch
from torch import nn

from .layers import EinsumLayer, Product
from .utils import SamplingContext


class Einet(nn.Module):
    def __init__(self, cfg, in_features):
        super().__init__()
        self.in_features = in_features
        self.in_channels = cfg.MODEL.EINET.K
        self.num_sums = cfg.MODEL.EINET.K
        self.depth = cfg.MODEL.EINET.D
        self.rand_perm = torch.tensor(np.random.permutation(self.in_features))

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
            )
            layers.append(l)
            _in_features = l._out_features
            _in_channels = self.num_sums

        self.layers = nn.ModuleList(layers)
        self.prod = Product(_in_features, cardinality=_in_features)

    def forward(self, log_p):
        # Forward through all layers, bottom up
        for layer in self.layers:
            log_p = layer(log_p)

        # Root layer merges all scopes that are left
        log_p = self.prod(log_p)
        return log_p

    def sample(self, z: torch.Tensor):
        context = SamplingContext(n=z.shape[0])
        context = self.prod.sample(context.n, context)
        for layer in reversed(self.layers):
            context = layer.sample(context.n, context)

        old_shape = z.shape
        z = z.reshape(z.shape[0], -1, z.shape[-1])
        index = context.parent_indices.unsqueeze(-1)
        z = z.gather(dim=2, index=index)

        z = z.view(*old_shape[:-1])
        return z
