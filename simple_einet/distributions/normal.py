from typing import Tuple

import torch
from simple_einet.distributions.abstract_leaf import AbstractLeaf
from simple_einet.type_checks import check_valid
from torch import distributions as dist
from torch import nn

from simple_einet.utils import SamplingContext


class Normal(AbstractLeaf):
    """Gaussian layer. Maps each input feature to its gaussian log likelihood."""

    def __init__(
        self,
        num_features: int,
        num_channels: int,
        num_leaves: int,
        num_repetitions: int,
    ):
        """Creat a gaussian layer.

        Args:
            out_channels: Number of parallel representations for each input feature.
            in_features: Number of input features.
            num_repetitions: Number of parallel repetitions of this layer.

        """
        super().__init__(num_features, num_channels, num_leaves, num_repetitions)

        # Create gaussian means and stds
        self.means = nn.Parameter(torch.randn(1, num_channels, num_features, num_leaves, num_repetitions))
        self.log_stds = nn.Parameter(torch.rand(1, num_channels, num_features, num_leaves, num_repetitions))

    def _get_base_distribution(self, context: SamplingContext = None):
        return dist.Normal(loc=self.means, scale=self.log_stds.exp())


class RatNormal(AbstractLeaf):
    """Implementation as in RAT-SPN

    Gaussian layer. Maps each input feature to its gaussian log likelihood."""

    def __init__(
        self,
        num_features: int,
        num_leaves: int,
        num_channels: int,
        num_repetitions: int = 1,
        min_sigma: float = 0.1,
        max_sigma: float = 1.0,
        min_mean: float = None,
        max_mean: float = None,
    ):
        """Creat a gaussian layer.

        Args:
            out_channels: Number of parallel representations for each input feature.
            in_features: Number of input features.

        """
        super().__init__(
            num_features=num_features,
            num_leaves=num_leaves,
            num_repetitions=num_repetitions,
            num_channels=num_channels,
        )

        # Create gaussian means and stds
        self.means = nn.Parameter(torch.randn(1, num_channels, num_features, num_leaves, num_repetitions))

        if min_sigma is not None and max_sigma is not None:
            # Init from normal
            self.stds = nn.Parameter(torch.randn(1, num_channels, num_features, num_leaves, num_repetitions))
        else:
            # Init uniform between 0 and 1
            self.stds = nn.Parameter(torch.rand(1, num_channels, num_features, num_leaves, num_repetitions))

        self.min_sigma = check_valid(min_sigma, float, 0.0, max_sigma)
        self.max_sigma = check_valid(max_sigma, float, min_sigma)
        self.min_mean = check_valid(min_mean, float, upper_bound=max_mean, allow_none=True)
        self.max_mean = check_valid(max_mean, float, min_mean, allow_none=True)

    def _get_base_distribution(self, context: SamplingContext = None) -> "CustomNormal":
        if self.min_sigma < self.max_sigma:
            sigma_ratio = torch.sigmoid(self.stds)
            sigma = self.min_sigma + (self.max_sigma - self.min_sigma) * sigma_ratio
        else:
            sigma = 1.0

        means = self.means
        if self.max_mean:
            assert self.min_mean is not None
            mean_range = self.max_mean - self.min_mean
            means = torch.sigmoid(self.means) * mean_range + self.min_mean

        # d = dist.Normal(means, sigma)
        d = CustomNormal(means, sigma)
        return d


class CustomNormal:
    """Basic Normal class that can sample given mu and sigma."""

    def __init__(self, mu: torch.Tensor, sigma: torch.Tensor):
        self.mu = mu
        self.sigma = sigma

    def sample(self, sample_shape: Tuple[int]):
        num_samples = sample_shape[0]
        eps = torch.randn((num_samples,) + self.mu.shape, dtype=self.mu.dtype, device=self.mu.device)
        samples = self.mu.unsqueeze(0) + self.sigma.unsqueeze(0) * eps
        return samples

    def log_prob(self, x):
        return dist.Normal(self.mu, self.sigma).log_prob(x)
