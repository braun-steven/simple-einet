from typing import List

import numpy as np
import torch

from simple_einet.abstract_layers import AbstractLayer
from simple_einet.layers.distributions import AbstractLeaf
from simple_einet.sampling_utils import SamplingContext, index_one_hot


class FactorizedLeaf(AbstractLayer):
    """
    A 'meta'-leaf layer that combines multiple scopes of a base-leaf layer via naive factorization.

    Attributes:
        num_features (int): Number of input features.
        num_features_out (int): Number of output features.
        num_repetitions (int): Number of repetitions.
        base_leaf (AbstractLeaf): The base leaf layer.
        scopes (torch.Tensor): The scopes of the factorized groups of RVs.
    """

    def __init__(
        self,
        num_features: int,
        num_features_out: int,
        num_repetitions,
        base_leaf: AbstractLeaf,
    ):
        """
        Args:
            num_features (int): Number of input features.
            num_features_out (int): Number of output features.
            num_repetitions (int): Number of repetitions.
            base_leaf (AbstractLeaf): The base leaf layer.
        """

        super().__init__(num_features, num_repetitions=num_repetitions)

        self.base_leaf = base_leaf
        self.num_features_out = num_features_out

        # Size of the factorized groups of RVs
        cardinality = int(np.floor(self.num_features / self.num_features_out))

        # Construct equal group sizes, such that (sum(group_sizes) == num_features) and the are num_features_out groups
        group_sizes = np.ones(self.num_features_out, dtype=int) * cardinality
        rest = self.num_features - cardinality * self.num_features_out
        for i in range(rest):
            group_sizes[i] += 1
        np.random.shuffle(group_sizes)

        # Construct mapping of scopes from in_features -> out_features
        scopes = torch.zeros(num_features, self.num_features_out, num_repetitions)
        for r in range(num_repetitions):
            idxs = torch.randperm(n=self.num_features)
            offset = 0
            for o in range(num_features_out):
                group_size = group_sizes[o]
                low = offset
                high = offset + group_size
                offset = high
                if o == num_features_out - 1:
                    high = self.num_features
                scopes[idxs[low:high], o, r] = 1

        self.register_buffer("scopes", scopes)

    def forward(self, x: torch.Tensor, marginalized_scopes: List[int]):
        """
        Forward pass through the factorized leaf layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_input_channels, num_leaves, num_repetitions).
            marginalized_scopes (List[int]): List of integers representing the marginalized scopes.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_output_channels, num_leaves, num_repetitions).
        """
        # Forward through base leaf
        x = self.base_leaf(x, marginalized_scopes)

        # Factorize input channels
        x = x.sum(dim=1)

        # Merge scopes by naive factorization
        x = torch.einsum("bicr,ior->bocr", x, self.scopes)

        assert x.shape == (
            x.shape[0],
            self.num_features_out,
            self.base_leaf.num_leaves,
            self.num_repetitions,
        )
        return x

    def sample(self, ctx: SamplingContext) -> torch.Tensor:
        """
        Samples the factorized leaf layer by generating `context.num_samples` samples from the base leaf layer,
        and then mapping them to the factorized leaf layer using the indices specified in the `context`
        argument. If `context.is_differentiable` is True, the mapping is done using one-hot indexing.

        Args:
            ctx (SamplingContext, optional): The sampling context to use. Defaults to None.

        Returns:
            torch.Tensor: A tensor of shape `(context.num_samples, self.num_features_out, self.num_leaves)`,
            representing the samples generated from the factorized leaf layer.
        """
        # Save original indices_out and set context indices_out to none, such that the out_channel
        # are not filtered in the base_leaf sampling procedure
        indices_out = ctx.indices_out
        ctx.indices_out = None
        samples = self.base_leaf.sample(ctx=ctx)

        # Check that shapes match as expected
        assert samples.shape == (
            ctx.num_samples,
            self.base_leaf.num_channels,
            self.base_leaf.num_features,
            self.base_leaf.num_leaves,
        )

        if ctx.is_differentiable:
            # Select the correct repetitions
            scopes = self.scopes.unsqueeze(0)  # make space for batch dim
            r_idx = ctx.indices_repetition.view(ctx.num_samples, 1, 1, -1)
            scopes = index_one_hot(scopes, index=r_idx, dim=-1)

            indices_in = index_one_hot(indices_out.unsqueeze(1), index=scopes.unsqueeze(-1), dim=2)
            indices_in = indices_in.unsqueeze(1)  # make space for channel dim
            samples = index_one_hot(samples, index=indices_in, dim=-1)
        else:
            # Select the correct repetitions
            scopes = self.scopes[..., ctx.indices_repetition].permute(2, 0, 1)
            rnge_in = torch.arange(self.num_features_out, device=samples.device)
            scopes = (scopes * rnge_in).sum(-1).long()
            indices_in_gather = indices_out.gather(dim=1, index=scopes)
            indices_in_gather = indices_in_gather.view(ctx.num_samples, 1, -1, 1)

            indices_in_gather = indices_in_gather.expand(-1, samples.shape[1], -1, -1)
            indices_in_gather = indices_in_gather.repeat(1, 1, self.base_leaf.cardinality, 1)
            samples = samples.gather(dim=-1, index=indices_in_gather)
            samples.squeeze_(-1)  # Remove num_leaves dimension

        return samples

    def extra_repr(self):
        return f"num_features={self.num_features}, num_features_out={self.num_features_out}"
