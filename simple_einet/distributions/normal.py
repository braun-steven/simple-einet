from typing import Tuple

import torch
from simple_einet.distributions.abstract_leaf import AbstractLeaf
from simple_einet.type_checks import check_valid
from torch import distributions as dist
from torch import nn

from simple_einet.sampling_utils import SamplingContext


class Normal(AbstractLeaf):
    """Gaussian layer. Maps each input feature to its gaussian log likelihood."""

    def __init__(
        self,
        num_features: int,
        num_channels: int,
        num_leaves: int,
        num_repetitions: int,
    ):
        """
        Initializes a Normal distribution with the given parameters.

        Args:
            num_features (int): The number of features in the input tensor.
            num_channels (int): The number of channels in the input tensor.
            num_leaves (int): The number of leaves in the tree structure.
            num_repetitions (int): The number of repetitions of the tree structure.
        """
        super().__init__(num_features, num_channels, num_leaves, num_repetitions)

        # Create gaussian means and stds
        self.means = nn.Parameter(torch.randn(1, num_channels, num_features, num_leaves, num_repetitions))
        self.log_stds = nn.Parameter(torch.rand(1, num_channels, num_features, num_leaves, num_repetitions))

    def _get_base_distribution(self, context: SamplingContext = None):
        return dist.Normal(loc=self.means, scale=self.log_stds.exp())


class RatNormal(AbstractLeaf):
    """Implementation as in RAT-SPN

    Gaussian layer. Maps each input feature to its gaussian log likelihood.

    Sigmas are constrained to be in min_sigma and max_sigma.
    Means are constrained to be in min_mean and max_mean.
    """

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
        """
        Initializes a Normal distribution with learnable parameters for the means and standard deviations.

        Args:
            num_features (int): The number of features in the input tensor.
            num_leaves (int): The number of leaves in the tree structure.
            num_channels (int): The number of channels in the input tensor.
            num_repetitions (int, optional): The number of repetitions for each feature. Defaults to 1.
            min_sigma (float, optional): The minimum value for the standard deviation. Defaults to 0.1.
            max_sigma (float, optional): The maximum value for the standard deviation. Defaults to 1.0.
            min_mean (float, optional): The minimum value for the mean. Defaults to None.
            max_mean (float, optional): The maximum value for the mean. Defaults to None.
        """
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
    """
    A custom implementation of the Normal distribution.

    This class allows to sample from a Normal distribution with mean `mu` and standard deviation `sigma`.
    The `sample` method returns a tensor of samples from the distribution, with shape `sample_shape + mu.shape`.
    The `log_prob` method returns the log probability density/mass function evaluated at `x`.

    Args:
        mu (torch.Tensor): The mean of the Normal distribution.
        sigma (torch.Tensor): The standard deviation of the Normal distribution.
    """
    def __init__(self, mu: torch.Tensor, sigma: torch.Tensor):
        self.mu = mu
        self.sigma = sigma

    def sample(self, sample_shape: Tuple[int]):
            """
            Generates random samples from the normal distribution with mean `mu` and standard deviation `sigma`.

            Args:
                sample_shape (Tuple[int]): The shape of the desired output tensor.

            Returns:
                samples (torch.Tensor): A tensor of shape `sample_shape` containing random samples from the normal distribution.
            """
            num_samples = sample_shape[0]
            eps = torch.randn((num_samples,) + self.mu.shape, dtype=self.mu.dtype, device=self.mu.device)
            samples = self.mu.unsqueeze(0) + self.sigma.unsqueeze(0) * eps
            return samples

    def log_prob(self, x):
        """
        Computes the log probability density of the normal distribution at the given value.

        Args:
            x (torch.Tensor): The value(s) at which to evaluate the log probability density.

        Returns:
            torch.Tensor: The log probability density of the normal distribution at the given value(s).
        """
        return dist.Normal(self.mu, self.sigma).log_prob(x)
