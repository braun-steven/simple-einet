from simple_einet.type_checks import check_valid


import logging
from abc import ABC, abstractmethod
from simple_einet.layers import AbstractLayer
from simple_einet.sampling_utils import SamplingContext, index_one_hot
from typing import List
from torch import distributions as dist, nn
import torch


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


def dist_mode(distribution: dist.Distribution, context: SamplingContext = None) -> torch.Tensor:
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
    from simple_einet.distributions.normal import CustomNormal
    from simple_einet.distributions.binomial import CustomBinomial

    if isinstance(distribution, CustomNormal):
        # Repeat the mode along the batch axis
        return distribution.mu.repeat(context.num_samples, 1, 1, 1, 1)
    elif isinstance(distribution, dist.Bernoulli):
        mode = distribution.probs.clone()
        mode[mode >= 0.5] = 1.0
        mode[mode < 0.5] = 0.0
        return mode.repeat(context.num_samples, 1, 1, 1, 1)
    elif isinstance(distribution, dist.Binomial) or isinstance(distribution, CustomBinomial):
        mode = distribution.probs.clone()
        total_count = distribution.total_count
        mode = torch.floor(mode * (total_count + 1))
        if mode.shape[0] == 1:
            return mode.repeat(context.num_samples, 1, 1, 1, 1)
        else:
            return mode

    else:
        raise Exception(f"MPE not yet implemented for type {type(distribution)}")


def dist_sample(distribution: dist.Distribution, context: SamplingContext = None) -> torch.Tensor:
    """
    Sample n samples from a given distribution.

    Args:
        distribution: Leaf distribution from which to sample from.
        context: Sampling context.

    Returns:
        torch.Tensor: Samples from the given distribution.
    """

    # Sample from the specified distribution
    if context.is_mpe or context.mpe_at_leaves:
        samples = dist_mode(distribution, context)
        samples = samples.unsqueeze(1)
    else:
        from simple_einet.distributions import CustomNormal

        if type(distribution) == dist.Normal:
            distribution = dist.Normal(loc=distribution.loc, scale=distribution.scale * context.temperature_leaves)
        elif type(distribution) == CustomNormal:
            distribution = CustomNormal(mu=distribution.mu, sigma=distribution.sigma * context.temperature_leaves)
        samples = distribution.sample(sample_shape=(context.num_samples,))

    assert (
        samples.shape[1] == 1
    ), "Something went wrong. First sample size dimension should be size 1 due to the distribution parameter dimensions. Please report this issue."

    # if not context.is_differentiable:  # This happens only in the non-differentiable context
    samples.squeeze_(1)
    num_samples, num_channels, num_features, num_leaves, num_repetitions = samples.shape

    # Index samples to get the correct repetitions
    # TODO: instead of indexing samples to get correct repetition, rather do this before sampling
    #  to save time
    if not context.is_differentiable:
        r_idxs = context.indices_repetition.view(-1, 1, 1, 1, 1)
        r_idxs = r_idxs.expand(-1, num_channels, num_features, num_leaves, -1)
        samples = samples.gather(dim=-1, index=r_idxs)
        samples = samples.squeeze(-1)
    else:
        r_idxs = context.indices_repetition.view(num_samples, 1, 1, 1, num_repetitions)
        samples = index_one_hot(samples, index=r_idxs, dim=-1)

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

    Attributes:
        num_features: Number of input features.
        num_channels: Number of input features.
        num_leaves: Number of parallel representations for each input feature.
        num_repetitions: Number of parallel repetitions of this layer.
        cardinality: Number of random variables covered by a single leaf.
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
        """
        Applies dropout to the input tensor `x` according to the dropout probability
        `self.dropout`. Dropout is only applied during training (when `model.train()`
        has been called).

        Args:
            x (torch.Tensor): The input tensor to apply dropout to.

        Returns:
            torch.Tensor: The input tensor with dropout applied.
        """
        if self.dropout > 0.0 and self.training:
            dropout_indices = self._bernoulli_dist.sample(
                x.shape,
            ).bool()
            x[dropout_indices] = 0.0
        return x

    def _marginalize_input(self, x: torch.Tensor, marginalized_scopes: List[int]) -> torch.Tensor:
        """
        Marginalizes the input tensor `x` along the dimensions specified in `marginalized_scopes`.

        Args:
            x (torch.Tensor): The input tensor to be marginalized.
            marginalized_scopes (List[int]): A list of dimensions to be marginalized.

        Returns:
            torch.Tensor: The marginalized tensor.
        """
        # Marginalize nans set by user
        if marginalized_scopes is not None:
            # Transform to tensor
            if type(marginalized_scopes) != torch.Tensor:
                s = torch.tensor(marginalized_scopes)
            else:
                s = marginalized_scopes

            # Adjust for leaf cardinality
            if self.cardinality > 1:
                s = marginalized_scopes.div(self.cardinality, rounding_mode="floor")

            x[:, :, s] = self.marginalization_constant
        return x

    def forward(self, x, marginalized_scopes: List[int]):
        """
        Forward pass through the distribution.

        Args:
            x (torch.Tensor): Input tensor.
            marginalized_scopes (List[int]): List of scopes to marginalize.

        Returns:
            torch.Tensor: Output tensor after marginalization.
        """
        # Forward through base distribution
        d = self._get_base_distribution()
        x = dist_forward(d, x)

        x = self._marginalize_input(x, marginalized_scopes)

        return x

    @abstractmethod
    def _get_base_distribution(self, context: SamplingContext = None) -> dist.Distribution:
        """Get the underlying torch distribution."""
        pass

    def sample(self, num_samples: int = None, context: SamplingContext = None) -> torch.Tensor:
        """
        Sample from the distribution represented by this leaf node.

        Args:
            num_samples (int, optional): The number of samples to draw from the distribution. If None, a single sample is drawn.
            context (SamplingContext, optional): The sampling context to use when drawing samples.

        Returns:
            torch.Tensor: A tensor of shape (num_samples,) or (1,) containing the drawn samples.
        """
        d = self._get_base_distribution(context)
        samples = dist_sample(distribution=d, context=context)
        return samples

    def extra_repr(self):
        return f"num_features={self.num_features}, num_leaves={self.num_leaves}, out_shape={self.out_shape}"
