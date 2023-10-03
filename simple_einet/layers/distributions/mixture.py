from typing import List

import torch
from torch import nn

from simple_einet.layers.distributions.abstract_leaf import AbstractLeaf
from simple_einet.layers.sum import SumLayer
from simple_einet.sampling_utils import SamplingContext


class Mixture(AbstractLeaf):
    def __init__(
        self,
        distributions,
        num_features: int,
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
            num_repetitions: Number of times to repeat the layer.
            dropout: Dropout probability.
        """
        super().__init__(num_features, out_channels, num_repetitions, dropout)
        # Build different layers for each distribution specified
        reprs = [distr(num_features, out_channels, num_repetitions, dropout) for distr in distributions]
        self.representations = nn.ModuleList(reprs)

        # Build sum layer as mixture of distributions
        self.sumlayer = SumLayer(
            num_features=num_features,
            num_sums_in=len(distributions) * out_channels,
            num_sums_out=out_channels,
            num_repetitions=num_repetitions,
        )

    def _get_base_distribution(self):
        raise Exception("Not implemented")

    def forward(self, x, marginalized_scopes: List[int]):
        """
        Forward pass of the Mixture layer.

        Args:
            x: Input tensor.
            marginalized_scopes: List of marginalized scopes.

        Returns:
            Output tensor.
        """
        results = [d(x) for d in self.representations]

        # Stack along output channel dimension
        x = torch.cat(results, dim=2)

        # Build mixture of different leafs per in_feature
        x = self.sumlayer(x)
        return x

    def sample(self, ctx: SamplingContext) -> torch.Tensor:
        """
        Sample from the Mixture layer.

        Args:
            ctx: Sampling context.

        Returns:
            Sampled tensor.
        """
        # Sample from sum mixture layer
        ctx = self.sumlayer.sample(ctx=ctx)

        # Collect samples from different distribution layers
        samples = []
        for d in self.representations:
            sample_d = d.sample(ctx=ctx)
            samples.append(sample_d)

        # Stack along channel dimension
        samples = torch.cat(samples, dim=2)

        # If parent index into out_channels are given
        if ctx.indices_out is not None:
            # Choose only specific samples for each feature/scope
            samples = torch.gather(samples, dim=2, index=ctx.indices_out.unsqueeze(-1)).squeeze(-1)

        return samples
