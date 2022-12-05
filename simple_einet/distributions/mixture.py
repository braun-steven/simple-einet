import logging
from abc import ABC, abstractmethod
from typing import Iterable, List, Tuple, Any, Dict

import numpy as np
import torch
from torch import distributions as dist
from torch import nn
from torch.nn import functional as F

from simple_einet.utils import SamplingContext, invert_permutation
from simple_einet.layers import AbstractLayer, Sum
from simple_einet.type_checks import check_valid
from simple_einet.distributions.abstract_leaf import AbstractLeaf, dist_mode


class Mixture(AbstractLeaf):
    def __init__(
        self,
        distributions,
        in_features: int,
        out_channels,
        num_repetitions,
        dropout=0.0,
    ):
        """
        Create a layer that stack multiple representations of a feature along the scope dimension.

        Args:
            distributions: List of possible distributions to represent the feature with.
            out_channels: out_channels of how many nodes each distribution is assigned to.
            in_features: Number of input features.
        """
        super().__init__(in_features, out_channels, num_repetitions, dropout)
        # Build different layers for each distribution specified
        reprs = [distr(in_features, out_channels, num_repetitions, dropout) for distr in distributions]
        self.representations = nn.ModuleList(reprs)

        # Build sum layer as mixture of distributions
        self.sumlayer = Sum(
            num_features=in_features,
            num_sums_in=len(distributions) * out_channels,
            num_sums_out=out_channels,
            num_repetitions=num_repetitions,
        )

    def _get_base_distribution(self):
        raise Exception("Not implemented")

    def forward(self, x, marginalized_scopes: List[int]):
        results = [d(x) for d in self.representations]

        # Stack along output channel dimension
        x = torch.cat(results, dim=2)

        # Build mixture of different leafs per in_feature
        x = self.sumlayer(x)
        return x

    def sample(self, num_samples: int = None, context: SamplingContext = None) -> torch.Tensor:
        # Sample from sum mixture layer
        context = self.sumlayer.sample(context=context)

        # Collect samples from different distribution layers
        samples = []
        for d in self.representations:
            sample_d = d.sample(context=context)
            samples.append(sample_d)

        # Stack along channel dimension
        samples = torch.cat(samples, dim=2)

        # If parent index into out_channels are given
        if context.indices_out is not None:
            # Choose only specific samples for each feature/scope
            samples = torch.gather(samples, dim=2, index=context.indices_out.unsqueeze(-1)).squeeze(-1)

        return samples
