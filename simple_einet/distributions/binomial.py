from torch import distributions as dist, nn
import numpy as np
import torchvision.models as models
from simple_einet.utils import SamplingContext
from typing import List, Tuple, Union
import torch
from torch import nn
from simple_einet.type_checks import check_valid

from simple_einet.distributions.abstract_leaf import (
    AbstractLeaf,
    dist_forward,
    dist_mode,
    dist_sample,
)
from torch.nn import functional as F


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
        self.probs = nn.Parameter(torch.randn(1, num_channels, num_features, num_leaves, num_repetitions))

    def _get_base_distribution(self, context: SamplingContext = None):
        # Use sigmoid to ensure, that probs are in valid range
        if context is not None and context.is_differentiable:
            return CustomBinomial(probs=self.probs.sigmoid(), total_count=self.total_count)
        else:
            return dist.Binomial(probs=self.probs.sigmoid(), total_count=self.total_count)


class CustomBinomial:
    def __init__(self, probs, total_count):
        self.probs = probs
        self.total_count = total_count

    def sample(self, sample_shape: Tuple[int]):
        # Normal approximation to be differentiable
        mu = self.total_count * self.probs
        sigma = mu * (1 - self.probs)

        num_samples = sample_shape[0]
        eps = torch.randn((num_samples,) + mu.shape, dtype=mu.dtype, device=mu.device)
        samples = mu.unsqueeze(0) + sigma.unsqueeze(0) * eps
        samples = samples.clip(0, self.total_count)
        return samples

    def log_prob(self, x):
        return dist.Binomial(probs=self.probs, total_count=self.total_count).log_prob(x)


class ConditionalBinomial(AbstractLeaf):
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
        # Get conditional input (TODO: make this flexible with an index array defined during construction)
        x_cond = x[:, :, self.cond_idxs, None, None]
        d = self.get_conditioned_distribution(x_cond)

        # Compute lls
        x = dist_forward(d, x)

        # Marginalize
        x = self._marginalize_input(x, marginalized_scopes)

        return x

    def sample(self, num_samples: int = None, context: SamplingContext = None) -> torch.Tensor:
        ev = context.evidence
        x_cond = ev[:, :, self.cond_idxs, None, None]
        d = self.get_conditioned_distribution(x_cond)

        # Sample from the specified distribution
        if context.is_mpe:
            samples = dist_mode(d, context)
        else:
            samples = d.sample()

        (
            num_samples,
            num_channels,
            num_features,
            num_leaves,
            num_repetitions,
        ) = samples.shape

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

    def _get_base_distribution(self) -> dist.Distribution:
        raise NotImplementedError("This should not happen.")
