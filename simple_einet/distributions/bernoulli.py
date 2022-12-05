import torch
from simple_einet.distributions.abstract_leaf import AbstractLeaf
from torch import distributions as dist
from torch import nn


class Bernoulli(AbstractLeaf):
    """Bernoulli layer. Maps each input feature to its gaussian log likelihood."""

    def __init__(self, num_features: int, num_channels: int, num_leaves: int, num_repetitions: int):
        """Creat a gaussian layer.

        Args:
            out_channels: Number of parallel representations for each input feature.
            in_features: Number of input features.
            num_repetitions: Number of parallel repetitions of this layer.

        """
        super().__init__(num_features, num_channels, num_leaves, num_repetitions)

        # Create bernoulli parameters
        self.probs = nn.Parameter(torch.randn(1, num_channels, num_features, num_leaves, num_repetitions))

    def _get_base_distribution(self):
        # Use sigmoid to ensure, that probs are in valid range
        return dist.Bernoulli(probs=self.sigmoid(self.probs))
