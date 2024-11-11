from typing import List, Tuple

import numpy as np
import torch
from torch import distributions as dist
from torch import nn
from torch.nn import functional as F

from simple_einet.layers.distributions.abstract_leaf import AbstractLeaf, dist_mode, dist_forward
from simple_einet.sampling_utils import SamplingContext
from simple_einet.type_checks import check_valid


class MultivariateNormal(AbstractLeaf):
    """Multivariate Gaussian layer."""

    def __init__(
        self,
        num_features: int,
        num_channels: int,
        num_leaves: int,
        cardinality: int,
        num_repetitions: int = 1,
    ):
        """Creat a multivariate gaussian layer.

        # NOTE: Marginalization in this leaf is not supported for now. Reason: The implementation parallelizes the
        # representation of the different multivariate gaussians for efficiency. That means we have num_dists x (K x K)
        # gaussians of cardinality K. If we were now to marginalize a single scope, we would need to pick the distributions
        # in which that scope appears and reduce these gaussians to size (K-1 x K-1) by deleting the appropriate elements
        # in their means and rows/columns in their covariance matrices. This results in non-parallelizable execution of
        # computations with the other non-marginalized gaussians.

        Args:
            num_features: Number of input features.
            num_channels: Number of input channels.
            num_leaves: Number of parallel representations for each input feature.
            num_repetitions: Number of parallel repetitions of this layer.
            cardinality: Number of features covered.

        """
        super().__init__(num_features, num_channels, num_leaves, num_repetitions, cardinality)
        self._pad_value = num_features % cardinality
        self.out_features = np.ceil(num_features / cardinality).astype(int)

        # Number of separate mulitvariate normal distributions, each covering #cardinality features
        param_shape = (num_channels, self.out_features, num_leaves, num_repetitions)
        self._num_dists = np.prod(param_shape)

        # Create gaussian means and covs
        self.means = nn.Parameter(torch.randn(*param_shape, cardinality))

        # Generate covariance matrix via the cholesky decomposition: s = L'L where L is a triangular matrix
        self.L_diag_log = nn.Parameter(torch.zeros(*param_shape, cardinality))
        self.L_offdiag = nn.Parameter(torch.tril(torch.zeros(*param_shape, cardinality, cardinality), diagonal=-1))

    @property
    def scale_tril(self):
        """Get the lower triangular matrix L of the cholesky decomposition of the covariance matrix."""
        L_diag = self.L_diag_log.exp()  # Ensure that L_diag is positive
        L_offdiag = self.L_offdiag.tril(-1)  # Take the off-diagonal part of L_offdiag
        L_full = torch.diag_embed(L_diag) + L_offdiag  # Construct full lower triangular matrix
        return L_full

    def _get_base_distribution(self, ctx: SamplingContext = None, marginalized_scopes=None):
        # View means and scale_tril
        means = self.means.view(self._num_dists, self.cardinality)
        scale_tril = self.scale_tril.view(self._num_dists, self.cardinality, self.cardinality)

        mv = CustomMultivariateNormalDist(
            mean=means,
            scale_tril=scale_tril,
            num_channels=self.num_channels,
            out_features=self.out_features,
            num_leaves=self.num_leaves,
            num_repetitions=self.num_repetitions,
            cardinality=self.cardinality,
            pad_value=self._pad_value,
            num_dists=self._num_dists,
            num_features=self.num_features,
        )
        return mv


class CustomMultivariateNormalDist:
    def __init__(
        self,
        mean,
        scale_tril,
        num_dists,
        num_channels,
        out_features,
        num_leaves,
        num_repetitions,
        cardinality,
        pad_value,
        num_features,
    ):
        self.mean = mean
        self.scale_tril = scale_tril
        self.num_channels = num_channels
        self.out_features = out_features
        self.num_leaves = num_leaves
        self.num_features = num_features
        self.num_repetitions = num_repetitions
        self.cardinality = cardinality
        self._pad_value = pad_value
        self._num_dists = num_dists

        self.mv = dist.MultivariateNormal(loc=self.mean, scale_tril=self.scale_tril)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        # Pad dummy variable via reflection
        if self._pad_value != 0:
            x = F.pad(x, pad=[0, 0, 0, self._pad_value], mode="reflect")

        # Split features into groups
        x = x.view(
            batch_size,
            self.num_channels,
            self.out_features,
            1,
            1,
            self.cardinality,
        )

        # Repeat groups by number of output_channels and number of repetitions
        x = x.repeat(1, 1, 1, self.num_leaves, self.num_repetitions, 1)

        # Merge groups and repetitions
        x = x.view(batch_size, self._num_dists, self.cardinality)

        # Compute multivariate gaussians
        x = self.mv.log_prob(x)
        x = x.view(x.shape[0], self.num_channels, self.out_features, self.num_leaves, self.num_repetitions)
        return x

    def sample(self, sample_shape: Tuple[int]) -> torch.Tensor:
        samples = self.mv.sample(sample_shape=sample_shape)

        samples = samples.view(
            sample_shape[0],
            self.num_channels,
            self.out_features,
            self.num_leaves,
            self.num_repetitions,
            self.cardinality,
        )

        samples = samples.permute(0, 1, 2, 5, 3, 4)
        samples = samples.reshape(
            sample_shape[0], self.num_channels, self.num_features, self.num_leaves, self.num_repetitions
        )

        # Add empty dimension to align shape with samples from uniform distributions such that this layer can be used in
        # the same way as other layers in dist_sample (abstract_leaf.py)
        samples.unsqueeze_(1)

        return samples

    def mpe(self, num_samples) -> torch.Tensor:
        """
        Generates MPE samples from this distribution.

        Args:
            num_samples (int): The number of samples to generate.

        Returns:
            samples (torch.Tensor):
        """
        samples = self.mean.repeat(num_samples, 1, 1, 1, 1)

        samples = samples.view(
            num_samples,
            self.num_channels,
            self.out_features,
            self.num_leaves,
            self.num_repetitions,
            self.cardinality,
        )

        samples = samples.permute(0, 1, 2, 5, 3, 4)
        samples = samples.reshape(
            num_samples, self.num_channels, self.num_features, self.num_leaves, self.num_repetitions
        )
        return samples
