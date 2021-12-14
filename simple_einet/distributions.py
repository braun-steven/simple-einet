"""
Module that contains a set of distributions with learnable parameters.
"""

import logging
from abc import abstractmethod
from typing import List

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
    # Make room for out_channels and num_repetitions of layer
    if x.dim() == 3:  # Number of repetition dimension already exists
        x = x.unsqueeze(2)  # Shape [n, d, 1, r]
    elif x.dim() == 2:
        x = x.unsqueeze(2).unsqueeze(3)  # Shape: [n, d, 1, 1]

    # Compute log-likelihodd
    x = distribution.log_prob(x)  # Shape: [n, d, oc, r]

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
    else:
        raise Exception(f"MPE not yet implemented for type {type(distribution)}")


def dist_sample(distribution: dist.Distribution, context: SamplingContext = None) -> torch.Tensor:
    """
    Sample n samples from a given distribution.

    Args:
        repetition_indices: Indices into the repetition axis.
        distribution (dists.Distribution): Base distribution to sample from.
        parent_indices (torch.Tensor): Tensor of indexes that point to specific representations of single features/scopes.
    """

    # Sample from the specified distribution
    if context.is_mpe:
        samples = _mode(distribution, context)
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
    num_samples, d, c, r = samples.shape

    # Filter each sample by its specific repetition
    tmp = torch.zeros(num_samples, d, c, device=context.repetition_indices.device)
    for i in range(num_samples):
        tmp[i, :, :] = samples[i, :, :, context.repetition_indices[i]]
    samples = tmp

    # If parent index into out_channels are given
    if context.parent_indices is not None:
        # Choose only specific samples for each feature/scope
        samples = torch.gather(samples, dim=2, index=context.parent_indices.unsqueeze(-1)).squeeze(
            -1
        )

    return samples


class Leaf(AbstractLayer):
    """
    Abstract layer that maps each input feature into a specified
    representation, e.g. Gaussians.

    Implementing layers shall be valid distributions.

    If the input at a specific position is NaN, the variable will be marginalized.
    """

    def __init__(self, in_features: int, out_channels: int, num_repetitions: int = 1, dropout=0.0, cardinality=1):
        """
        Create the leaf layer.

        Args:
            in_features: Number of input features.
            out_channels: Number of parallel representations for each input feature.
            num_repetitions: Number of parallel repetitions of this layer.
            dropout: Dropout probability.
            cardinality: Number of random variables covered by a single leaf.
        """
        super().__init__(in_features=in_features, num_repetitions=num_repetitions)
        self.in_features = check_valid(in_features, int, 1)
        self.out_channels = check_valid(out_channels, int, 1)
        self.num_repetitions = check_valid(num_repetitions, int, 1)
        self.cardinality = check_valid(cardinality, int, 1)
        dropout = check_valid(dropout, float, 0.0, 1.0)
        self.dropout = nn.Parameter(torch.tensor(dropout), requires_grad=False)

        self.out_features = in_features
        self.out_shape = f"(N, {in_features}, {out_channels})"

        # Marginalization constant
        self.marginalization_constant = nn.Parameter(torch.zeros(1), requires_grad=False)

        # Dropout bernoulli
        self._bernoulli_dist = torch.distributions.Bernoulli(probs=self.dropout)

    def _apply_dropout(self, x: torch.Tensor) -> torch.Tensor:
        # Apply dropout sampled from a bernoulli during training (model.train() has been called)
        if self.dropout > 0.0 and self.training:
            dropout_indices = self._bernoulli_dist.sample(x.shape).bool()
            x[dropout_indices] = 0.0
        return x

    def _marginalize_input(self, x: torch.Tensor, marginalized_scopes: List[int]) -> torch.Tensor:
        # Marginalize nans set by user
        if marginalized_scopes:
            s = list(set(torch.tensor(marginalized_scopes).div(self.cardinality, rounding_mode="floor").tolist()))
            x[:, s] = self.marginalization_constant
        return x

    def forward(self, x, marginalized_scopes: List[int]):
        # Forward through base distribution
        d = self._get_base_distribution()
        x = dist_forward(d, x)

        x = self._marginalize_input(x, marginalized_scopes)
        x = self._apply_dropout(x)

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
        return f"{self.__class__.__name__}(in_features={self.in_features}, out_channels={self.out_channels}, dropout={self.dropout}, out_shape={self.out_shape})"


class Normal(Leaf):
    """Gaussian layer. Maps each input feature to its gaussian log likelihood."""

    def __init__(self, in_features: int, out_channels: int, num_repetitions: int = 1, dropout=0.0):
        """Creat a gaussian layer.

        Args:
            out_channels: Number of parallel representations for each input feature.
            in_features: Number of input features.
            num_repetitions: Number of parallel repetitions of this layer.

        """
        super().__init__(in_features, out_channels, num_repetitions, dropout)

        # Create gaussian means and stds
        self.means = nn.Parameter(torch.randn(1, in_features, out_channels, num_repetitions))
        self.stds = nn.Parameter(torch.rand(1, in_features, out_channels, num_repetitions))
        self.gauss = dist.Normal(loc=self.means, scale=self.stds)

    def _get_base_distribution(self):
        return self.gauss


class Bernoulli(Leaf):
    """Bernoulli layer. Maps each input feature to its gaussian log likelihood."""

    def __init__(self, in_features: int, out_channels: int, num_repetitions: int = 1, dropout=0.0):
        """Creat a gaussian layer.

        Args:
            out_channels: Number of parallel representations for each input feature.
            in_features: Number of input features.
            num_repetitions: Number of parallel repetitions of this layer.

        """
        super().__init__(in_features, out_channels, num_repetitions, dropout)

        # Create bernoulli parameters
        self.probs = nn.Parameter(torch.randn(1, in_features, out_channels, num_repetitions))

    def _get_base_distribution(self):
        # Use sigmoid to ensure, that probs are in valid range
        probs_ratio = torch.sigmoid(self.probs)
        return dist.Bernoulli(probs=probs_ratio)


class Binomial(Leaf):
    def __init__(
        self,
        in_features: int,
        out_channels: int,
        total_count: int,
        num_repetitions: int = 1,
        dropout=0,
    ):
        super().__init__(
            in_features, out_channels, num_repetitions=num_repetitions, dropout=dropout
        )

        self.total_count = check_valid(total_count, int, lower_bound=1)

        # Create binomial parameters
        self.probs = nn.Parameter(torch.rand(1, in_features, out_channels, num_repetitions))
        # Learnable s
        self.sigmoid_scale = nn.Parameter(torch.tensor(2.0))

    def _get_base_distribution(self):
        # Use sigmoid to ensure, that probs are in valid range
        return dist.Binomial(self.total_count, probs=torch.sigmoid(self.probs * self.sigmoid_scale))


class MultivariateNormal(Leaf):
    """Multivariate Gaussian layer."""

    def __init__(
        self,
        in_features: int,
        out_channels: int,
        cardinality: int,
        num_repetitions: int = 1,
        dropout=0.0,
        min_sigma: float = 0.1,
        max_sigma: float = 1.0,
    ):
        """Creat a multivariate gaussian layer.

        Args:
            out_channels: Number of parallel representations for each input feature.
            in_features: Number of input features.
            num_repetitions: Number of parallel repetitions of this layer.
            cardinality: Number of features covered.

        """
        # TODO: Fix for num_repetitions
        super().__init__(in_features, out_channels, num_repetitions, dropout, cardinality)
        self._pad_value = in_features % cardinality
        self.out_features = np.ceil(in_features / cardinality).astype(int)
        self._n_dists = np.ceil(in_features / cardinality).astype(int)

        self.min_sigma = check_valid(min_sigma, float, 0.0, max_sigma)
        self.max_sigma = check_valid(max_sigma, float, min_sigma)

        # Create gaussian means and covs
        self.means = nn.Parameter(
            torch.randn(out_channels * self._n_dists * self.num_repetitions, cardinality)
        )

        # Generate covariance matrix via the cholesky decomposition: s = A'A where A is a triangular matrix
        # Further ensure, that diag(a) > 0 everywhere, such that A has full rank
        rand = torch.zeros(
            out_channels * self._n_dists * self.num_repetitions, cardinality, cardinality
        )

        for i in range(cardinality):
            rand[:, i, i] = 1.0

        rand = rand + torch.randn_like(rand) * 1e-1

        # Make matrices triangular and remove diagonal entries
        cov_tril_wo_diag = rand.tril(diagonal=-1)
        cov_tril_wi_diag = torch.rand(out_channels * self._n_dists * self.num_repetitions, cardinality, cardinality)

        self.cov_tril_wo_diag = nn.Parameter(cov_tril_wo_diag)
        self.cov_tril_wi_diag = nn.Parameter(cov_tril_wi_diag)
        # self._mv = dist.MultivariateNormal(loc=self.means, scale_tril=self.triangular)
        # Reassign means since mv __init__ creates a copy and thus would loose track for autograd
        # self._mv.loc.requires_grad_(True)
        # self.means = nn.Parameter(self._mv.loc)

        self.out_shape = f"(N, {self.out_features}, {self.out_channels})"

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
        x = x.repeat(1, self.num_repetitions, self.out_channels, 1, 1)  #  [n, r, oc, d/k, k]

        # Merge groups and repetitions
        x = x.view(
            batch_size, self.num_repetitions * self.out_channels * self._n_dists, self.cardinality
        )  #  [n, r * d/k * oc, k]

        # Compute multivariate gaussians
        # Output shape: [n, out_channels, d / cardinality]
        mv = self._get_base_distribution()
        x = mv.log_prob(x)  #  [n, r * d/k * oc]
        x = x.view(
            batch_size, self.num_repetitions, self.out_channels, self._n_dists
        )  # [n, r, oc, d/k]
        x = x.permute(0, 3, 2, 1)  # [n, d/k, oc, r]

        # Marginalize and apply dropout
        x = self._marginalize_input(x, marginalized_scopes)
        x = self._apply_dropout(x)

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
                self.out_channels,
                self._n_dists,
                self.cardinality,
            )

        num_samples, num_repetitions, out_channels, n_dists, cardinality = samples.shape

        # Filter each sample by its specific repetition
        tmp = torch.zeros(
            num_samples,
            out_channels,
            n_dists,
            cardinality,
            device=context.repetition_indices.device,
        )
        for i in range(num_samples):
            tmp[i] = samples[i, context.repetition_indices[i], ...]

        samples = tmp  # [n, oc, d/k, k]

        samples = samples.view(
            context.num_samples,
            self.out_channels,
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


class Beta(Leaf):
    """Beta layer. Maps each input feature to its beta log likelihood."""

    def __init__(self, in_features: int, out_channels: int, num_repetitions: int = 1, dropout=0.0):
        """Creat a beta layer.

        Args:
            out_channels: Number of parallel representations for each input feature.
            in_features: Number of input features.
            num_repetitions: Number of parallel repetitions of this layer.

        """
        super().__init__(in_features, out_channels, num_repetitions, dropout)

        # Create beta parameters
        self.concentration0 = nn.Parameter(
            torch.rand(1, in_features, out_channels, num_repetitions)
        )
        self.concentration1 = nn.Parameter(
            torch.rand(1, in_features, out_channels, num_repetitions)
        )
        self.beta = dist.Beta(
            concentration0=self.concentration0, concentration1=self.concentration1
        )

    def _get_base_distribution(self):
        return self.beta


class Cauchy(Leaf):
    """Cauchy layer. Maps each input feature to cauchy beta log likelihood."""

    def __init__(self, in_features: int, out_channels: int, num_repetitions: int = 1, dropout=0.0):
        """Creat a cauchy layer.

        Args:
            out_channels: Number of parallel representations for each input feature.
            in_features: Number of input features.
            num_repetitions: Number of parallel repetitions of this layer.

        """
        super().__init__(in_features, out_channels, num_repetitions, dropout)
        self.means = nn.Parameter(torch.randn(1, in_features, out_channels, num_repetitions))
        self.stds = nn.Parameter(torch.rand(1, in_features, out_channels, num_repetitions))
        self.cauchy = dist.Cauchy(loc=self.means, scale=self.stds)

    def _get_base_distribution(self):
        return self.cauchy


class Chi2(Leaf):
    """Chi square distribution layer"""

    def __init__(self, in_features: int, out_channels: int, num_repetitions: int = 1, dropout=0.0):
        """Creat a chi square layer.

        Args:
            out_channels: Number of parallel representations for each input feature.
            in_features: Number of input features.
            num_repetitions: Number of parallel repetitions of this layer.

        """
        super().__init__(in_features, out_channels, num_repetitions, dropout)
        self.df = nn.Parameter(torch.rand(1, in_features, out_channels, num_repetitions))
        self.chi2 = dist.Chi2(df=self.df)

    def _get_base_distribution(self):
        return self.chi2


class Mixture(Leaf):
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
        reprs = [
            distr(in_features, out_channels, num_repetitions, dropout) for distr in distributions
        ]
        self.representations = nn.ModuleList(reprs)

        # Build sum layer as mixture of distributions
        self.sumlayer = Sum(
            in_features=in_features,
            in_channels=len(distributions) * out_channels,
            out_channels=out_channels,
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
        if context.parent_indices is not None:
            # Choose only specific samples for each feature/scope
            samples = torch.gather(
                samples, dim=2, index=context.parent_indices.unsqueeze(-1)
            ).squeeze(-1)

        return samples


class IsotropicMultivariateNormal(Leaf):
    """Isotropic multivariate gaussian layer.

    The covariance is simplified to:

    cov = sigma^2 * I

    Maps k input feature to their multivariate gaussian log likelihood."""

    def __init__(self, in_features, out_channels, num_repetitions, cardinality, dropout=0.0):
        """Creat a gaussian layer.

        Args:
            out_channels: Number of parallel representations for each input feature.
            cardinality: Number of features per gaussian.
            in_features: Number of input features.

        """
        super().__init__(in_features, out_channels, num_repetitions, dropout)
        self.cardinality = cardinality

        # Number of different distributions: total number of features
        # divided by the number of features in each gaussian

        self._pad_value = in_features % cardinality
        self._out_features = np.ceil(in_features / cardinality).astype(int)

        self._n_dists = np.ceil(in_features / cardinality).astype(int)

        # Create gaussian means and stds
        self.means = nn.Parameter(
            torch.randn(out_channels, self._n_dists, cardinality, num_repetitions)
        )
        self.stds = nn.Parameter(
            torch.rand(out_channels, self._n_dists, cardinality, num_repetitions)
        )
        self.cov_factors = nn.Parameter(
            torch.zeros(out_channels, self._n_dists, cardinality, num_repetitions),
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

        # Make room for out_channels of layer
        # Output shape: [n, 1, d]
        batch_size = x.shape[0]
        x = x.reshape(batch_size, 1, self._n_dists, self.cardinality)

        # Compute multivariate gaussians
        # Output shape: [n, out_channels, d / cardinality]
        x = self.gauss.log_prob(x)

        # Output shape: [n, d / cardinality, out_channels]
        x = x.permute((0, 2, 1))

        x = self._marginalize_input(x)
        x = self._apply_dropout(x)

        return x

    def sample(self, n=None, context: SamplingContext = None) -> torch.Tensor:
        """TODO: Multivariate need special treatment."""
        raise Exception("Not yet implemented")

    def _get_base_distribution(self):
        return self.gauss


class Gamma(Leaf):
    """Gamma distribution layer."""

    def __init__(self, in_features: int, out_channels: int, num_repetitions: int = 1, dropout=0.0):
        """Creat a gamma layer.

        Args:
            out_channels: Number of parallel representations for each input feature.
            in_features: Number of input features.

        """
        super().__init__(in_features, out_channels, num_repetitions, dropout)
        self.concentration = nn.Parameter(torch.rand(1, in_features, out_channels, num_repetitions))
        self.rate = nn.Parameter(torch.rand(1, in_features, out_channels, num_repetitions))
        self.gamma = dist.Gamma(concentration=self.concentration, rate=self.rate)

    def _get_base_distribution(self):
        return self.gamma


class Poisson(Leaf):
    """Poisson distribution layer."""

    def __init__(self, in_features: int, out_channels: int, num_repetitions: int = 1, dropout=0.0):
        """Creat a poisson layer.

        Args:
            out_channels: Number of parallel representations for each input feature.
            in_features: Number of input features.

        """
        super().__init__(in_features, out_channels, num_repetitions, dropout)
        self.rate = nn.Parameter(torch.rand(1, in_features, out_channels, num_repetitions))
        self.poisson = dist.Poisson(rate=self.rate)

    def _get_base_distribution(self):
        return self.poisson


class RatNormal(Leaf):
    """Implementation as in RAT-SPN

    Gaussian layer. Maps each input feature to its gaussian log likelihood."""

    def __init__(
        self,
        in_features: int,
        out_channels: int,
        num_repetitions: int = 1,
        dropout: float = 0.0,
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
        super().__init__(in_features, out_channels, num_repetitions, dropout)

        # Create gaussian means and stds
        self.means = nn.Parameter(torch.randn(1, in_features, out_channels, num_repetitions))

        if min_sigma is not None and max_sigma is not None:
            # Init from normal
            self.stds = nn.Parameter(torch.randn(1, in_features, out_channels, num_repetitions))
        else:
            # Init uniform between 0 and 1
            self.stds = nn.Parameter(torch.rand(1, in_features, out_channels, num_repetitions))

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
