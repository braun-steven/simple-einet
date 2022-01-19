"""
Module that contains a set of distributions with learnable parameters.
"""

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

logger = logging.getLogger(__name__)


def dist_forward(distribution, x: torch.Tensor):
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

    if x.dim() == 3:  # [N, C, D]
        x = x.unsqueeze(-1).unsqueeze(-1)  # [N, C, D, 1, 1]

    # Compute log-likelihodd
    try:
        x = distribution.log_prob(x)  # Shape: [n, d, oc, r]
    except ValueError as e:
        print("min:", x.min())
        print("max:", x.max())
        raise e

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
        mode = distribution.probs.clone()
        total_count = distribution.total_count
        mode = torch.floor(mode * (total_count + 1))
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
        samples = samples.unsqueeze(1)
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
    num_samples, num_channels, num_features, num_leaves, num_repetitions = samples.shape

    # Index samples to get the correct repetitions
    r_idxs = context.indices_repetition.view(-1, 1, 1, 1, 1)
    r_idxs = r_idxs.expand(-1, num_channels, num_features, num_leaves, -1)
    samples = samples.gather(dim=-1, index=r_idxs)
    samples = samples.squeeze(-1)

    # If parent index into out_channels are given
    if context.indices_out is not None:
        # Choose only specific samples for each feature/scope
        samples = torch.gather(samples, dim=2, index=context.indices_out.unsqueeze(-1)).squeeze(-1)

    return samples


class AbstractLeaf(AbstractLayer, ABC):
    """
    Abstract layer that maps each input feature into a specified
    representation, e.g. Gaussians.

    Implementing layers shall be valid distributions.
    """

    def __init__(
        self,
        num_features: int,
        num_channels: int,
        num_leaves: int,
        num_repetitions: int = 1,
        cardinality=1,
    ):
        """
        Create the leaf layer.

        Args:
            num_features: Number of input features.
            num_channels: Number of input features.
            num_leaves: Number of parallel representations for each input feature.
            num_repetitions: Number of parallel repetitions of this layer.
            cardinality: Number of random variables covered by a single leaf.
        """
        super().__init__(num_features=num_features, num_repetitions=num_repetitions)
        self.num_features = check_valid(num_features, int, 1)
        self.num_channels = check_valid(num_channels, int, 1)
        self.num_leaves = check_valid(num_leaves, int, 1)
        self.num_repetitions = check_valid(num_repetitions, int, 1)
        self.cardinality = check_valid(cardinality, int, 1)

        self.out_features = num_features
        self.out_shape = f"(N, {num_features}, {num_leaves})"

        # Marginalization constant
        self.marginalization_constant = nn.Parameter(torch.zeros(1), requires_grad=False)

    def _apply_dropout(self, x: torch.Tensor) -> torch.Tensor:
        # Apply dropout sampled from a bernoulli during training (model.train() has been called)
        if self.dropout > 0.0 and self.training:
            dropout_indices = self._bernoulli_dist.sample(x.shape).bool()
            x[dropout_indices] = 0.0
        return x

    def _marginalize_input(self, x: torch.Tensor, marginalized_scopes: List[int]) -> torch.Tensor:
        # Marginalize nans set by user
        if marginalized_scopes is not None:
            if type(marginalized_scopes) != torch.Tensor:
                marginalized_scopes = torch.tensor(marginalized_scopes)
            s = marginalized_scopes.div(self.cardinality, rounding_mode="floor")
            s = list(set(s.tolist()))
            x[:, :, s] = self.marginalization_constant
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

    def extra_repr(self):
        return f"num_features={self.num_features}, num_leaves={self.num_leaves}, out_shape={self.out_shape}"


class Normal(AbstractLeaf):
    """Gaussian layer. Maps each input feature to its gaussian log likelihood."""

    def __init__(self, num_features: int, num_channels: int, num_leaves: int, num_repetitions: int):
        """Creat a gaussian layer.

        Args:
            out_channels: Number of parallel representations for each input feature.
            in_features: Number of input features.
            num_repetitions: Number of parallel repetitions of this layer.

        """
        super().__init__(num_features, num_channels, num_leaves, num_repetitions)

        # Create gaussian means and stds
        self.means = nn.Parameter(
            torch.randn(1, num_channels, num_features, num_leaves, num_repetitions)
        )
        self.stds = nn.Parameter(
            torch.rand(1, num_channels, num_features, num_leaves, num_repetitions)
        )
        self.gauss = dist.Normal(loc=self.means, scale=self.stds)

    def _get_base_distribution(self):
        return self.gauss


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
        self.probs = nn.Parameter(
            torch.randn(1, num_channels, num_features, num_leaves, num_repetitions)
        )

    def _get_base_distribution(self):
        # Use sigmoid to ensure, that probs are in valid range
        return dist.Bernoulli(probs=self.sigmoid(self.probs))


class Binomial(AbstractLeaf):
    def __init__(
        self,
        num_features: int,
        num_channels: int,
        num_leaves: int,
        num_repetitions: int,
        total_count: int,
    ):
        super().__init__(
            num_features=num_features,
            num_channels=num_channels,
            num_leaves=num_leaves,
            num_repetitions=num_repetitions,
        )

        self.total_count = check_valid(total_count, int, lower_bound=1)

        # Create binomial parameters
        self.probs = nn.Parameter(
            0.5 + torch.rand(1, num_channels, num_features, num_leaves, num_repetitions) * 0.1
        )

    def _get_base_distribution(self):
        # Use sigmoid to ensure, that probs are in valid range
        return dist.Binomial(self.total_count, probs=torch.sigmoid(self.probs))


class MultiDistributionLayer(AbstractLeaf):
    def __init__(
        self,
        scopes_to_dist: List[Tuple[Iterable[int], AbstractLeaf, Dict[str, Any]]],
        num_features: int,
        num_channels: int,
        num_leaves: int,
        num_repetitions: int = 1,
    ):
        """Construct a leaf layer that represents multiple distributions.

        Args:
            scopes_to_dist (List[Tuple[Iterable[int], AbstractLeaf]]): List of lists of indices that represent scopes to a
                specific distribution (AbstractLeaf object).
            num_features (int): Number of input features.
            num_channels (int): Number of input channels.
            num_leaves (int): Number of leaves per distribution per input random variable.
            num_repetitions (int, optional): Number of repetitions. Defaults to 1.
        """
        super().__init__(
            num_features,
            num_channels,
            num_leaves,
            num_repetitions=num_repetitions,
        )

        # Check that the index list covers all features
        all_scopes = []
        for (scopes, _, _) in scopes_to_dist:
            for scope in scopes:
                all_scopes.append(scope)

        assert len(all_scopes) == num_features
        scope_list = []
        dists = []
        for (scopes, dist_class, dist_kwargs) in scopes_to_dist:
            # Construct distribution object
            dist = dist_class(
                num_features=len(scopes),
                num_channels=num_channels,
                num_leaves=num_leaves,
                num_repetitions=num_repetitions,
                **dist_kwargs,
            )
            scope_list.append(scopes)
            dists.append(dist)

        self.scopes = scope_list
        self.dists = nn.ModuleList(dists)

        self.scopes_to_dist = scopes_to_dist
        self.inverted_index = invert_permutation(torch.tensor(all_scopes))

        # Instantiate distributions

        # Check if scopes are already sorted, if so, inversion is not necessary in the forward pass
        self.needs_inversion = all_scopes != list(sorted(all_scopes))

    def forward(self, x, marginalized_scopes: List[int] = None):

        # Collect lls from all distributions
        lls_all = []

        # Forward through all base distributions
        for (scope, dist) in zip(self.scopes, self.dists):
            x_d = x[:, :, scope]
            lls = dist(x_d, marginalized_scopes=None)
            lls_all.append(lls)

        # Stack along feature dimension
        lls = torch.cat(lls_all, dim=2)

        # If inversion is necessary, permute features to obtain the original order
        if self.needs_inversion:
            lls = lls[:, :, self.inverted_index]

        # Marginalize
        lls = self._marginalize_input(lls, marginalized_scopes)

        return lls

    def sample(self, num_samples: int = None, context: SamplingContext = None) -> torch.Tensor:

        all_samples = []
        for (scope, dist) in zip(self.scopes, self.dists):
            samples = dist.sample(num_samples, context)
            all_samples.append(samples)

        samples = torch.cat(all_samples, dim=2)

        # If inversion is necessary, permute features to obtain the original order
        if self.needs_inversion:
            samples = samples[:, :, self.inverted_index]

        return samples

    def _get_base_distribution(self) -> dist.Distribution:
        raise NotImplementedError(
            "MultiDistributionLayer does not implement _get_base_distribution."
        )


class MultivariateNormal(AbstractLeaf):
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
        raise NotImplementedError(
            "MultivariateNormal has not been adapted to the new implementation yet - sorry."
        )
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
        cov_tril_wi_diag = torch.rand(
            out_channels * self._n_dists * self.num_repetitions, cardinality, cardinality
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
        x = x.repeat(1, self.num_repetitions, self.num_leaves, 1, 1)  #  [n, r, oc, d/k, k]

        # Merge groups and repetitions
        x = x.view(
            batch_size, self.num_repetitions * self.num_leaves * self._n_dists, self.cardinality
        )  #  [n, r * d/k * oc, k]

        # Compute multivariate gaussians
        # Output shape: [n, out_channels, d / cardinality]
        mv = self._get_base_distribution()
        x = mv.log_prob(x)  #  [n, r * d/k * oc]
        x = x.view(
            batch_size, self.num_repetitions, self.num_leaves, self._n_dists
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
                self.num_leaves,
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
        reprs = [
            distr(in_features, out_channels, num_repetitions, dropout) for distr in distributions
        ]
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
            samples = torch.gather(samples, dim=2, index=context.indices_out.unsqueeze(-1)).squeeze(
                -1
            )

        return samples


class RatNormal(AbstractLeaf):
    """Implementation as in RAT-SPN

    Gaussian layer. Maps each input feature to its gaussian log likelihood."""

    def __init__(
        self,
        num_features: int,
        num_leaves: int,
        num_channels: int,
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
        super().__init__(
            num_features=num_features,
            num_leaves=num_leaves,
            num_repetitions=num_repetitions,
            num_channels=num_channels,
        )

        # Create gaussian means and stds
        self.means = nn.Parameter(
            torch.randn(1, num_channels, num_features, num_leaves, num_repetitions)
        )

        if min_sigma is not None and max_sigma is not None:
            # Init from normal
            self.stds = nn.Parameter(
                torch.randn(1, num_channels, num_features, num_leaves, num_repetitions)
            )
        else:
            # Init uniform between 0 and 1
            self.stds = nn.Parameter(
                torch.rand(1, num_channels, num_features, num_leaves, num_repetitions)
            )

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


if __name__ == "__main__":
    leaf = MultiDistributionLayer(
        scopes_to_dist=[
            (2 + torch.arange(2), Binomial, {"total_count": 2}),
            (torch.arange(2), RatNormal, {"min_sigma": 1e-3, "max_sigma": 1.0}),
            (4 + torch.arange(4), RatNormal, {"min_sigma": 1e-3, "max_sigma": 1.0}),
        ],
        num_features=8,
        num_channels=1,
        num_leaves=2,
        num_repetitions=1,
    )

    x = torch.randint(low=0, high=2, size=(1, 1, 8))
    leaf(x)
    print(leaf.sample(context=SamplingContext(1, indices_repetition=torch.zeros(1).long())))
