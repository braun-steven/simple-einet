"""
Module that contains a set of distributions with learnable parameters.
"""

import logging
from abc import abstractmethod
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch import distributions as dist
from torch import nn
from torch.nn import functional as F

from .utils import SamplingContext
from .layers import AbstractLayer, Sum
from .type_checks import check_valid

logger = logging.getLogger(__name__)


def dist_forward(distribution, x):
    """
    Forward pass with an arbitrary PyTorch distribution.

    Args:
        distribution: PyTorch base distribution which is used to compute the log probabilities of x.
        x: Input to compute the log probabilities of.
           Shape [n, d].

    Returns:
        torch.Tensor: Log probabilities for each feature.
    """
    # Make room for num_leaves and num_repetitions of layer
    if x.dim() == 4:  # [n, c, h, w]
        x = x.unsqueeze(-1).unsqueeze(-1)  # Shape [n, c, h, w, os=1, r=1]

    # Compute log-likelihodd
    x = distribution.log_prob(x)  # Shape: [n, c, h, w, os, r]

    return x


def _mode(distribution: dist.Distribution, context: SamplingContext = None) -> torch.Tensor:
    """
    Get the mode of a given distribution.

    Args:
        distribution: Leaf distribution from which to choose the mode from.
        context: Sampling context.
    Returns:
        torch.Tensor: Mode of the given distribution.
    """
    # TODO: Implement more torch distributions
    if isinstance(distribution, dist.Normal):
        # Repeat the mode along the batch axis
        return distribution.mean.repeat(context.num_samples, 1, 1, 1, 1)
    elif isinstance(distribution, dist.Bernoulli):
        mode = distribution.probs.clone()
        mode[mode >= 0.5] = 1.0
        mode[mode < 0.5] = 0.0
        return mode.repeat(context.num_samples, 1, 1, 1, 1)
    elif isinstance(distribution, dist.Binomial):
        probs = distribution.probs.clone()
        total_count = distribution.total_count
        mode = torch.floor(probs * (total_count + 1))
        return mode.repeat(context.num_samples, 1, 1, 1, 1)
    else:
        raise Exception(f"MPE not yet implemented for type {type(distribution)}")


def dist_sample(distribution: dist.Distribution, context: SamplingContext = None) -> torch.Tensor:
    """
    Sample n samples from a given distribution.

    Args:
        indices_repetition: Indices into the repetition axis.
        distribution (dists.Distribution): Base distribution to sample from.
        indices_out (torch.Tensor): Tensor of indexes that point to specific representations of single features/scopes.
    """

    # Sample from the specified distribution
    if context.is_mpe:
        samples = _mode(distribution, context)

        assert (
            samples.shape[0] == 1
        ), "Something went wrong. First sample size dimension should be size 1 due to the distribution parameter dimensions. Please report this issue."
    else:
        if type(distribution) == dist.Normal:
            distribution = dist.Normal(
                loc=distribution.loc, scale=distribution.scale * context.temperature_leaves
            )
        samples = distribution.sample(sample_shape=(context.num_samples,))

        assert (
            samples.shape[1] == 1
        ), "Something went wrong. First sample size dimension should be size 1 due to the distribution parameter dimensions. Please report this issue."
        samples.squeeze_(1)

    num_samples, height, width, num_leaves, num_repetitions = samples.shape

    r_idxs = context.indices_repetition.view(-1, 1, 1, 1, 1)
    r_idxs = r_idxs.expand(-1, height, width, num_leaves, -1)
    samples = samples.gather(dim=-1, index=r_idxs)
    samples = samples.squeeze(-1)

    # If parent index into num_leaves are given
    if context.indices_out is not None:
        # TODO: This has to be adapted to the width/height split

        # Choose only specific samples for each feature/scope
        samples = torch.gather(
            samples, dim=3, index=context.indices_out.unsqueeze(-1).unsqueeze(-1)
        ).squeeze(-1)

    return samples


class AbstractLeaf(AbstractLayer):
    """
    Abstract layer that maps each input feature into a specified
    representation, e.g. Gaussians.

    Implementing layers shall be valid distributions.

    If the input at a specific position is NaN, the variable will be marginalized.
    """

    def __init__(
        self,
        in_shape: Tuple[int, int],
        num_leaves: int,
        num_repetitions: int = 1,
        cardinality=1,
    ):
        """
        Create the leaf layer.

        Args:
            in_features: Number of input features.
            num_leaves: Number of parallel representations for each input feature.
            num_repetitions: Number of parallel repetitions of this layer.
            cardinality: Number of random variables covered by a single leaf.
        """
        super().__init__(in_shape=in_shape, num_repetitions=num_repetitions)
        self.num_leaves = check_valid(num_leaves, int, 1)
        self.num_repetitions = check_valid(num_repetitions, int, 1)
        self.cardinality = check_valid(cardinality, int, 1)

        self.out_shape = self.in_shape
        self.out_features = self.num_features

        # Marginalization constant
        self.marginalization_constant = nn.Parameter(torch.zeros(1), requires_grad=False)

    def _marginalize_input(self, x: torch.Tensor, marginalized_scopes: List[int]) -> torch.Tensor:
        # Marginalize nans set by user
        if marginalized_scopes:
            s = torch.tensor(marginalized_scopes).div(self.cardinality, rounding_mode="floor")
            s = list(set(s.tolist()))
            x[:, s] = self.marginalization_constant
        return x

    def forward(self, x, marginalized_scopes: List[int]):
        # Forward through base distribution
        d = self._get_base_distribution()
        x = dist_forward(d, x)

        x = self._marginalize_input(x, marginalized_scopes)

        return x

    @abstractmethod
    def _get_base_distribution(self) -> dist.Distribution:
        """Get the underlying torch distribution."""
        pass

    def sample(self, num_samples: int = None, context: SamplingContext = None) -> torch.Tensor:
        """
        Perform sampling, given indices from the parent layer that indicate which of the multiple representations
        for each input shall be used.
        """
        d = self._get_base_distribution()
        samples = dist_sample(distribution=d, context=context)
        return samples

    def __repr__(self):
        return f"{self.__class__.__name__}(in_shape={self.in_shape}, out_shape={self.out_shape}, num_leaves={self.num_leaves})"


class Normal(AbstractLeaf):
    """Gaussian layer. Maps each input feature to its gaussian log likelihood."""

    def __init__(self, in_features: int, num_leaves: int, num_repetitions: int = 1):
        """Creat a gaussian layer.

        Args:
            num_leaves: Number of parallel representations for each input feature.
            in_features: Number of input features.
            num_repetitions: Number of parallel repetitions of this layer.

        """
        super().__init__(in_features, num_leaves, num_repetitions)

        # Create gaussian means and stds
        self.means = nn.Parameter(torch.randn(1, in_features, num_leaves, num_repetitions))
        self.stds = nn.Parameter(torch.rand(1, in_features, num_leaves, num_repetitions))
        self.gauss = dist.Normal(loc=self.means, scale=self.stds)

    def _get_base_distribution(self):
        return self.gauss


class Bernoulli(AbstractLeaf):
    """Bernoulli layer. Maps each input feature to its gaussian log likelihood."""

    def __init__(self, in_features: int, num_leaves: int, num_repetitions: int = 1):
        """Creat a gaussian layer.

        Args:
            num_leaves: Number of parallel representations for each input feature.
            in_features: Number of input features.
            num_repetitions: Number of parallel repetitions of this layer.

        """
        super().__init__(in_features, num_leaves, num_repetitions)

        # Create bernoulli parameters
        self.probs = nn.Parameter(torch.randn(1, in_features, num_leaves, num_repetitions))

    def _get_base_distribution(self):
        # Use sigmoid to ensure, that probs are in valid range
        probs_ratio = torch.sigmoid(self.probs)
        return dist.Bernoulli(probs=probs_ratio)


class Binomial(AbstractLeaf):
    def __init__(
        self,
        in_shape: Tuple[int, int],
        num_leaves: int,
        total_count: int,
        num_repetitions: int = 1,
    ):
        super().__init__(in_shape, num_leaves, num_repetitions=num_repetitions)

        self.total_count = check_valid(total_count, int, lower_bound=1)

        # Create binomial parameters
        self.probs = nn.Parameter(torch.rand(1, *self.in_shape, num_leaves, num_repetitions))
        # Learnable s
        self.sigmoid_scale = nn.Parameter(torch.tensor(1.0))

    def _get_base_distribution(self):
        # Use sigmoid to ensure, that probs are in valid range
        return dist.Binomial(self.total_count, probs=torch.sigmoid(self.probs * self.sigmoid_scale))


class MultivariateNormal(AbstractLeaf):
    """Multivariate Gaussian layer."""

    def __init__(
        self,
        in_features: int,
        num_leaves: int,
        cardinality: int,
        num_repetitions: int = 1,
        min_sigma: float = 0.1,
        max_sigma: float = 1.0,
    ):
        """Creat a multivariate gaussian layer.

        Args:
            num_leaves: Number of parallel representations for each input feature.
            in_features: Number of input features.
            num_repetitions: Number of parallel repetitions of this layer.
            cardinality: Number of features covered.

        """
        # TODO: Fix for num_repetitions
        super().__init__(in_features, num_leaves, num_repetitions, cardinality)
        self._pad_value = in_features % cardinality
        self.out_features = np.ceil(in_features / cardinality).astype(int)
        self._n_dists = np.ceil(in_features / cardinality).astype(int)

        self.min_sigma = check_valid(min_sigma, float, 0.0, max_sigma)
        self.max_sigma = check_valid(max_sigma, float, min_sigma)

        # Create gaussian means and covs
        self.means = nn.Parameter(
            torch.randn(num_leaves * self._n_dists * self.num_repetitions, cardinality)
        )

        # Generate covariance matrix via the cholesky decomposition: s = A'A where A is a triangular matrix
        # Further ensure, that diag(a) > 0 everywhere, such that A has full rank
        rand = torch.zeros(
            num_leaves * self._n_dists * self.num_repetitions, cardinality, cardinality
        )

        for i in range(cardinality):
            rand[:, i, i] = 1.0

        rand = rand + torch.randn_like(rand) * 1e-1

        # Make matrices triangular and remove diagonal entries
        cov_tril_wo_diag = rand.tril(diagonal=-1)
        cov_tril_wi_diag = torch.rand(
            num_leaves * self._n_dists * self.num_repetitions, cardinality, cardinality
        )

        self.cov_tril_wo_diag = nn.Parameter(cov_tril_wo_diag)
        self.cov_tril_wi_diag = nn.Parameter(cov_tril_wi_diag)
        # self._mv = dist.MultivariateNormal(loc=self.means, scale_tril=self.triangular)
        # Reassign means since mv __init__ creates a copy and thus would loose track for autograd
        # self._mv.loc.requires_grad_(True)
        # self.means = nn.Parameter(self._mv.loc)

        self.out_shape = f"(N, {self.out_features}, {self.num_leaves})"

    def forward(self, x: torch.Tensor, marginalized_scopes: List[int]) -> torch.Tensor:
        batch_size = x.shape[0]

        # Pad dummy variable via reflection
        if self._pad_value != 0:
            x = F.pad(x, pad=[0, 0, 0, self._pad_value], mode="reflect")

        # Make room for repetitions: [n, 1, d]
        x = x.unsqueeze(1)

        # Split features into groups
        x = x.view(batch_size, 1, 1, self._n_dists, self.cardinality)  # [n, 1, 1, d/k, k]

        # Repeat groups by number of output_channels and number of repetitions
        x = x.expand(-1, self.num_repetitions, self.num_leaves, -1, -1)  #  [n, r, oc, d/k, k]

        # Merge groups and repetitions
        x = x.view(
            batch_size, self.num_repetitions * self.num_leaves * self._n_dists, self.cardinality
        )  #  [n, r * d/k * oc, k]

        # Compute multivariate gaussians
        # Output shape: [n, num_leaves, d / cardinality]
        mv = self._get_base_distribution()
        x = mv.log_prob(x)  #  [n, r * d/k * oc]
        x = x.view(
            batch_size, self.num_repetitions, self.num_leaves, self._n_dists
        )  # [n, r, oc, d/k]
        x = x.permute(0, 3, 2, 1)  # [n, d/k, oc, r]

        # Marginalize
        x = self._marginalize_input(x, marginalized_scopes)

        return x

    def sample(self, num_samples: int = None, context: SamplingContext = None) -> torch.Tensor:
        mv = self._get_base_distribution()

        # Sample from the specified distribution
        if context.is_mpe:
            samples = _mode(mv, context)
        else:
            samples = mv.sample(sample_shape=(context.num_samples,))

            samples = samples.view(
                context.num_samples,
                self.num_repetitions,
                self.num_leaves,
                self._n_dists,
                self.cardinality,
            )

        num_samples, num_repetitions, num_leaves, n_dists, cardinality = samples.shape

        # Filter each sample by its specific repetition
        tmp = torch.zeros(
            num_samples,
            num_leaves,
            n_dists,
            cardinality,
            device=context.indices_repetition.device,
        )
        for i in range(num_samples):
            tmp[i] = samples[i, context.indices_repetition[i], ...]

        samples = tmp  # [n, oc, d/k, k]

        samples = samples.view(
            context.num_samples,
            self.num_leaves,
            self._n_dists,
            self.cardinality,
        )
        samples = samples.permute(0, 2, 3, 1)

        return samples

    def _get_base_distribution(self):

        if self.min_sigma < self.max_sigma:
            # scale diag to [min_sigma, max_sigma]
            cov_diag = self.cov_tril_wi_diag
            sigma_ratio = torch.sigmoid(cov_diag)
            cov_diag = self.min_sigma + (self.max_sigma - self.min_sigma) * sigma_ratio
            cov_diag = cov_diag.tril()

            # scale tril to [-max_sigma, max_sigma]
            cov_tril = self.cov_tril_wo_diag
            sigma_ratio = torch.sigmoid(cov_tril)
            cov_tril = -1 * self.max_sigma + 2 * self.max_sigma * sigma_ratio
            cov_tril = cov_tril.tril(-1)

        else:
            cov_tril = self.cov_tril_wo_diag.tril(-1)
            cov_diag = self.cov_tril_wi_diag.tril().sigmoid()

        scale_tril = cov_tril + cov_diag
        mv = dist.MultivariateNormal(loc=self.means, scale_tril=scale_tril)

        # ic(cov_diag.mean(0))
        # ic(cov_tril.mean(0))
        return mv


class Beta(AbstractLeaf):
    """Beta layer. Maps each input feature to its beta log likelihood."""

    def __init__(self, in_features: int, num_leaves: int, num_repetitions: int = 1):
        """Creat a beta layer.

        Args:
            num_leaves: Number of parallel representations for each input feature.
            in_features: Number of input features.
            num_repetitions: Number of parallel repetitions of this layer.

        """
        super().__init__(in_features, num_leaves, num_repetitions)

        # Create beta parameters
        self.concentration0 = nn.Parameter(torch.rand(1, in_features, num_leaves, num_repetitions))
        self.concentration1 = nn.Parameter(torch.rand(1, in_features, num_leaves, num_repetitions))
        self.beta = dist.Beta(
            concentration0=self.concentration0, concentration1=self.concentration1
        )

    def _get_base_distribution(self):
        return self.beta


class Cauchy(AbstractLeaf):
    """Cauchy layer. Maps each input feature to cauchy beta log likelihood."""

    def __init__(self, in_features: int, num_leaves: int, num_repetitions: int = 1):
        """Creat a cauchy layer.

        Args:
            num_leaves: Number of parallel representations for each input feature.
            in_features: Number of input features.
            num_repetitions: Number of parallel repetitions of this layer.

        """
        super().__init__(in_features, num_leaves, num_repetitions)
        self.means = nn.Parameter(torch.randn(1, in_features, num_leaves, num_repetitions))
        self.stds = nn.Parameter(torch.rand(1, in_features, num_leaves, num_repetitions))
        self.cauchy = dist.Cauchy(loc=self.means, scale=self.stds)

    def _get_base_distribution(self):
        return self.cauchy


class Chi2(AbstractLeaf):
    """Chi square distribution layer"""

    def __init__(self, in_features: int, num_leaves: int, num_repetitions: int = 1):
        """Creat a chi square layer.

        Args:
            num_leaves: Number of parallel representations for each input feature.
            in_features: Number of input features.
            num_repetitions: Number of parallel repetitions of this layer.

        """
        super().__init__(in_features, num_leaves, num_repetitions)
        self.df = nn.Parameter(torch.rand(1, in_features, num_leaves, num_repetitions))
        self.chi2 = dist.Chi2(df=self.df)

    def _get_base_distribution(self):
        return self.chi2


class Mixture(AbstractLeaf):
    def __init__(
        self,
        distributions,
        in_features: int,
        num_leaves,
        num_repetitions,
    ):
        """
        Create a layer that stack multiple representations of a feature along the scope dimension.

        Args:
            distributions: List of possible distributions to represent the feature with.
            num_leaves: num_leaves of how many nodes each distribution is assigned to.
            in_features: Number of input features.
        """
        super().__init__(in_features, num_leaves, num_repetitions)
        # Build different layers for each distribution specified
        reprs = [distr(in_features, num_leaves, num_repetitions) for distr in distributions]
        self.representations = nn.ModuleList(reprs)

        # Build sum layer as mixture of distributions
        self.sumlayer = Sum(
            in_features=in_features,
            num_sums_in=len(distributions) * num_leaves,
            num_leaves=num_leaves,
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

        # If parent index into num_leaves are given
        if context.indices_out is not None:
            # Choose only specific samples for each feature/scope
            samples = torch.gather(samples, dim=2, index=context.indices_out.unsqueeze(-1)).squeeze(
                -1
            )

        return samples


class IsotropicMultivariateNormal(AbstractLeaf):
    """Isotropic multivariate gaussian layer.

    The covariance is simplified to:

    cov = sigma^2 * I

    Maps k input feature to their multivariate gaussian log likelihood."""

    def __init__(self, in_features, num_leaves, num_repetitions, cardinality):
        """Creat a gaussian layer.

        Args:
            num_leaves: Number of parallel representations for each input feature.
            cardinality: Number of features per gaussian.
            in_features: Number of input features.

        """
        super().__init__(in_features, num_leaves, num_repetitions)
        self.cardinality = cardinality

        # Number of different distributions: total number of features
        # divided by the number of features in each gaussian

        self._pad_value = in_features % cardinality
        self._out_features = np.ceil(in_features / cardinality).astype(int)

        self._n_dists = np.ceil(in_features / cardinality).astype(int)

        # Create gaussian means and stds
        self.means = nn.Parameter(
            torch.randn(num_leaves, self._n_dists, cardinality, num_repetitions)
        )
        self.stds = nn.Parameter(
            torch.rand(num_leaves, self._n_dists, cardinality, num_repetitions)
        )
        self.cov_factors = nn.Parameter(
            torch.zeros(num_leaves, self._n_dists, cardinality, num_repetitions),
            requires_grad=False,
        )
        self.gauss = dist.LowRankMultivariateNormal(
            loc=self.means, cov_factor=self.cov_factors, cov_diag=self.stds
        )

    def forward(self, x, marginalized_scopes: List[int]):
        # TODO: Fix for num_repetitions

        # Pad dummy variable via reflection
        if self._pad_value != 0:
            # Do unsqueeze and squeeze due to padding not being allowed on 2D tensors
            x = x.unsqueeze(1)
            x = F.pad(x, pad=[0, self._pad_value // 2], mode="reflect")
            x = x.squeeze(1)

        # Make room for num_leaves of layer
        # Output shape: [n, 1, d]
        batch_size = x.shape[0]
        x = x.reshape(batch_size, 1, self._n_dists, self.cardinality)

        # Compute multivariate gaussians
        # Output shape: [n, num_leaves, d / cardinality]
        x = self.gauss.log_prob(x)

        # Output shape: [n, d / cardinality, num_leaves]
        x = x.permute((0, 2, 1))

        x = self._marginalize_input(x)

        return x

    def sample(self, n=None, context: SamplingContext = None) -> torch.Tensor:
        """TODO: Multivariate need special treatment."""
        raise Exception("Not yet implemented")

    def _get_base_distribution(self):
        return self.gauss


class Gamma(AbstractLeaf):
    """Gamma distribution layer."""

    def __init__(self, in_features: int, num_leaves: int, num_repetitions: int = 1):
        """Creat a gamma layer.

        Args:
            num_leaves: Number of parallel representations for each input feature.
            in_features: Number of input features.

        """
        super().__init__(in_features, num_leaves, num_repetitions)
        self.concentration = nn.Parameter(torch.rand(1, in_features, num_leaves, num_repetitions))
        self.rate = nn.Parameter(torch.rand(1, in_features, num_leaves, num_repetitions))
        self.gamma = dist.Gamma(concentration=self.concentration, rate=self.rate)

    def _get_base_distribution(self):
        return self.gamma


class Poisson(AbstractLeaf):
    """Poisson distribution layer."""

    def __init__(self, in_features: int, num_leaves: int, num_repetitions: int = 1):
        """Creat a poisson layer.

        Args:
            num_leaves: Number of parallel representations for each input feature.
            in_features: Number of input features.

        """
        super().__init__(in_features, num_leaves, num_repetitions)
        self.rate = nn.Parameter(torch.rand(1, in_features, num_leaves, num_repetitions))
        self.poisson = dist.Poisson(rate=self.rate)

    def _get_base_distribution(self):
        return self.poisson


class RatNormal(AbstractLeaf):
    """Implementation as in RAT-SPN

    Gaussian layer. Maps each input feature to its gaussian log likelihood."""

    def __init__(
        self,
        in_shape: Tuple[int, int],
        num_leaves: int,
        num_repetitions: int = 1,
        min_sigma: float = 0.1,
        max_sigma: float = 1.0,
        min_mean: float = None,
        max_mean: float = None,
    ):
        """Creat a gaussian layer.

        Args:
            num_leaves: Number of parallel representations for each input feature.
            in_features: Number of input features.

        """
        super().__init__(in_shape, num_leaves, num_repetitions)

        # Create gaussian means and stds
        self.means = nn.Parameter(torch.randn(1, *in_shape, num_leaves, num_repetitions))

        if min_sigma is not None and max_sigma is not None:
            # Init from normal
            self.stds = nn.Parameter(torch.randn(1, *in_shape, num_leaves, num_repetitions))
        else:
            # Init uniform between 0 and 1
            self.stds = nn.Parameter(torch.rand(1, *in_shape, num_leaves, num_repetitions))

        self.min_sigma = check_valid(min_sigma, float, 0.0, max_sigma)
        self.max_sigma = check_valid(max_sigma, float, min_sigma)
        self.min_mean = check_valid(min_mean, float, upper_bound=max_mean, allow_none=True)
        self.max_mean = check_valid(max_mean, float, min_mean, allow_none=True)

    def _get_base_distribution(self) -> torch.distributions.Distribution:
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

        gauss = dist.Normal(means, torch.sqrt(sigma))
        return gauss


def truncated_normal_(tensor, mean=0, std=0.1):
    """
    Truncated normal from https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/15
    """
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
