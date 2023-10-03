from typing import List, Tuple, Union

import numpy as np
import torch
from torch import distributions as dist
from torch import nn

from simple_einet.layers.distributions.abstract_leaf import (
    AbstractLeaf,
    dist_forward,
    dist_mode,
)
from simple_einet.sampling_utils import SamplingContext
from simple_einet.type_checks import check_valid


class Binomial(AbstractLeaf):
    """Binomial layer. Maps each input feature to its binomial log likelihood."""

    def __init__(
        self,
        num_features: int,
        num_channels: int,
        num_leaves: int,
        num_repetitions: int,
        total_count: int,
    ):
        """
        Initializes a Binomial distribution with the given parameters.

        Args:
            num_features (int): The number of features in the input data.
            num_channels (int): The number of channels in the input data.
            num_leaves (int): The number of leaves in the tree.
            num_repetitions (int): The number of repetitions for each leaf.
            total_count (int): The total number of trials for the Binomial distribution.
        """
        super().__init__(
            num_features=num_features,
            num_channels=num_channels,
            num_leaves=num_leaves,
            num_repetitions=num_repetitions,
        )

        self.total_count = check_valid(total_count, int, lower_bound=1)

        # Create binomial parameters as unnormalized log probabilities
        self.logits = nn.Parameter(torch.randn(1, num_channels, num_features, num_leaves, num_repetitions))

    def _get_base_distribution(self, ctx: SamplingContext = None):
        # Use sigmoid to ensure, that probs are in valid range
        probs = self.logits.sigmoid()
        if ctx is not None and ctx.is_differentiable:
            return DifferentiableBinomial(probs=probs, total_count=self.total_count)
        else:
            return dist.Binomial(probs=probs, total_count=self.total_count)


class DifferentiableBinomial:
    """
    A custom implementation of the Binomial distribution, with differentiable sampling.

    Args:
        probs (torch.Tensor): The probability of success for each trial. Should have shape (batch_size,).
        total_count (int): The total number of trials.

    Attributes:
        probs (torch.Tensor): The probability of success for each trial. Should have shape (batch_size,).
        total_count (int): The total number of trials.
    """

    def __init__(self, probs, total_count):
        self.probs = probs
        self.total_count = total_count

    def sample(self, sample_shape: Tuple[int]):
        """
        Draws samples from the distribution using a gaussian differentiable approximation.


        # NOTE: The following would be the correct differentiable approximation using gumbel/simple trick
        # but since total_count can be quite large we use the normal approximation instead which doesn't need
        # to sample `total_count` times.
        # ```
        # p = self.probs
        # log_p = p.log()
        # log_1_p = (1 - p).log()
        # log_p = log_p.unsqueeze(-1).expand([*log_p.size(), self.total_count])
        # log_1_p = log_1_p.unsqueeze(-1).expand([*log_1_p.size(), self.total_count])
        # logits = torch.stack([log_1_p, log_p], dim=-1)
        # logits = logits.expand([*sample_shape, *logits.size()])
        # samples = SIMPLE(logits, dim=-1)
        # samples = (samples * torch.arange(0, 2).float()).sum([-2, -1])
        # return samples
        # ```

        Args:
            sample_shape (Tuple[int]): The shape of the desired sample.

        Returns:
            torch.Tensor: A tensor of shape (sample_shape[0], batch_size), containing the drawn samples.
        """
        mu = self.total_count * self.probs
        sigma = torch.sqrt(self.total_count * self.probs * (1 - self.probs))
        epsilon = torch.randn(sample_shape + mu.shape, device=mu.device)
        samples = mu + sigma * epsilon
        samples.clip_(min=0, max=self.total_count)
        return samples

    def log_prob(self, x):
        """
        Computes the log-probability of a given value under the distribution.

        Args:
            x (torch.Tensor): The value(s) for which to compute the log-probability. Should have shape (batch_size,).

        Returns:
            torch.Tensor: A tensor of shape (batch_size,), containing the log-probabilities.
        """
        return dist.Binomial(probs=self.probs, total_count=self.total_count).log_prob(x)


class ConditionalBinomial(AbstractLeaf):
    """
    A class representing a conditional binomial distribution.

    Allows a conditional function to be used to condition the binomial distribution.

    Args:
        num_features (int): The number of features in the input tensor.
        num_channels (int): The number of channels in the input tensor.
        num_leaves (int): The number of leaves in the tree.
        num_repetitions (int): The number of repetitions.
        total_count (int): The total count of the binomial distribution.
        cond_fn (nn.Module): The module used to condition the binomial distribution.
        cond_idxs (Union[List[int], torch.Tensor]): The indices of the conditioned input.

    Attributes:
        total_count (int): The total count of the binomial distribution.
        cond_fn (nn.Module): The module used to condition the binomial distribution.
        cond_idxs (Union[List[int], torch.Tensor]): The indices of the conditioned input.
        probs_conditioned_base (nn.Parameter): The base parameters for the conditioned binomial distribution.
        probs_unconditioned (nn.Parameter): The parameters for the unconditioned binomial distribution.
    """

    def __init__(
        self,
        num_features: int,
        num_channels: int,
        num_leaves: int,
        num_repetitions: int,
        total_count: int,
        cond_fn: nn.Module,
        cond_idxs: Union[List[int], torch.Tensor],
    ):
        """
        Initializes the ConditionalBinomial class.
        """
        super().__init__(
            num_features=num_features,
            num_channels=num_channels,
            num_leaves=num_leaves,
            num_repetitions=num_repetitions,
        )

        self.total_count = check_valid(total_count, int, lower_bound=1)
        self.cond_fn = cond_fn
        self.cond_idxs = cond_idxs

        self.probs_conditioned_base = nn.Parameter(
            0.5 + torch.rand(1, num_channels, num_features // 2, num_leaves, num_repetitions) * 0.1
        )
        self.probs_unconditioned = nn.Parameter(
            0.5 + torch.rand(1, num_channels, num_features // 2, num_leaves, num_repetitions) * 0.1
        )

    def get_conditioned_distribution(self, x_cond: torch.Tensor):
        """
        Get the conditioned torch distribution. That is, use `x_cond` to generate the paramters for the binomial
        distribution.

        Args:
            x_cond: Input condition.

        Returns:
            Parameters for the binomial distribution.
        """
        hw = int(np.sqrt(x_cond.shape[2]))
        assert hw * hw == x_cond.shape[2], "Input was not square."
        x_cond_shape = x_cond.shape

        # Get conditioned parameters
        probs_cond = self.cond_fn(x_cond.view(-1, x_cond.shape[1], hw, hw))
        probs_cond = probs_cond.view(
            x_cond_shape[0],
            x_cond_shape[1],
            self.num_leaves,
            self.num_repetitions,
            hw * hw,
        )
        probs_cond = probs_cond.permute(0, 1, 4, 2, 3)

        # Add conditioned parameters to default parameters
        probs_cond = self.probs_conditioned_base + probs_cond

        probs_unc = self.probs_unconditioned.expand(x_cond.shape[0], -1, -1, -1, -1)
        probs = torch.cat((probs_cond, probs_unc), dim=2)
        d = dist.Binomial(self.total_count, logits=probs)
        return d

    def forward(self, x, marginalized_scopes: List[int]):
        """
        Computes the forward pass of the ConditionalBinomial class.

        Args:
            x (torch.Tensor): The input tensor.
            marginalized_scopes (List[int]): The marginalized scopes.

        Returns:
            The output tensor.
        """
        # Get conditional input (TODO: make this flexible with an index array defined during construction)
        x_cond = x[:, :, self.cond_idxs, None, None]
        d = self.get_conditioned_distribution(x_cond)

        # Compute lls
        x = dist_forward(d, x)

        # Marginalize
        x = self._marginalize_input(x, marginalized_scopes)

        return x

    def sample(self, ctx: SamplingContext) -> torch.Tensor:
        """
        Samples from the ConditionalBinomial distribution.

        Args:
            ctx (SamplingContext): The sampling context.

        Returns:
            The generated samples.
        """
        ev = ctx.evidence
        x_cond = ev[:, :, self.cond_idxs, None, None]
        d = self.get_conditioned_distribution(x_cond)

        # Sample from the specified distribution
        if ctx.is_mpe:
            samples = dist_mode(d, ctx)
        else:
            samples = d.sample()

        # Index samples to get the correct repetitions
        r_idxs = ctx.indices_repetition.view(-1, 1, 1, 1, 1)
        r_idxs = r_idxs.expand(-1, self.num_channels, self.num_features, self.num_leaves, -1)
        samples = samples.gather(dim=-1, index=r_idxs)
        samples = samples.squeeze(-1)

        # If parent index into out_channels are given
        if ctx.indices_out is not None:
            # Choose only specific samples for each feature/scope
            samples = torch.gather(samples, dim=2, index=ctx.indices_out.unsqueeze(-1)).squeeze(-1)

        return samples

    def _get_base_distribution(self) -> dist.Distribution:
        """
        Gets the base distribution.

        Returns:
            The base distribution.
        """
        raise NotImplementedError("This should not happen.")
