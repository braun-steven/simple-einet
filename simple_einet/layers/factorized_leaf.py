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

        if self.num_features == self.num_features_out:
            return x

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

        # If return_leaf_params is True, we return the parameters of the leaf distribution
        # instead of the samples themselves
        if ctx.return_leaf_params:
            params = self.base_leaf.get_params()
            params = self._index_leaf_params(ctx, indices_out, params=params)
            return params
        else:
            samples = self.base_leaf.sample(ctx)
            samples = self._index_leaf_samples(ctx, indices_out, samples)
            return samples

    def _index_leaf_samples(self, ctx, indices_out, samples):
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

            indices_in_gather = indices_in_gather.expand(-1, self.base_leaf.num_channels, -1, -1)
            indices_in_gather = indices_in_gather.repeat(1, 1, self.base_leaf.cardinality, 1)
            samples = samples.gather(dim=-1, index=indices_in_gather)
            samples.squeeze_(-1)  # Remove num_leaves dimension
        return samples

    def _index_leaf_params(self, ctx, indices_out, params):
        """
        Same as _index_leaf_samples, but indexes the parameters of the leaf distribution instead of the samples.
        """
        num_params = params.shape[-1]  # Number of parameters, e.g. 2 for Normal (mu and sigma)
        if ctx.is_differentiable:
            r_idxs = ctx.indices_repetition.view(ctx.num_samples, 1, 1, 1, ctx.num_repetitions, 1)
            params = index_one_hot(params, index=r_idxs, dim=-2)  # -2 is num_repetitions dim

            # Select the correct repetitions
            scopes = self.scopes.unsqueeze(0)  # make space for batch dim
            r_idx = ctx.indices_repetition.view(ctx.num_samples, 1, 1, -1)
            scopes = index_one_hot(scopes, index=r_idx, dim=-1)

            indices_in = index_one_hot(indices_out.unsqueeze(1), index=scopes.unsqueeze(-1), dim=2)
            indices_in = indices_in.view(
                ctx.num_samples, 1, self.num_features, self.base_leaf.num_leaves, 1
            )  # make space for channel dim
            params = index_one_hot(params, index=indices_in, dim=-2)  # -2 is num_leaves dim
        else:
            # Filter for repetition
            r_idxs = ctx.indices_repetition.view(-1, 1, 1, 1, 1, 1)
            r_idxs = r_idxs.expand(
                -1, self.base_leaf.num_channels, self.num_features, self.base_leaf.num_leaves, -1, num_params
            )
            params = params.expand(ctx.num_samples, -1, -1, -1, -1, -1)
            params = params.gather(dim=-2, index=r_idxs)  # Repetition dim is -2, (-1 is param stack dim)
            params = params.squeeze(-2)  # Remove repetition dim
            # params is now [batch_size, num_channels, num_features, num_leaves, num_params]

            # Select the correct repetitions
            scopes = self.scopes[..., ctx.indices_repetition].permute(2, 0, 1)
            rnge_in = torch.arange(self.num_features_out, device=params.device)
            scopes = (scopes * rnge_in).sum(-1).long()
            indices_in_gather = indices_out.gather(dim=1, index=scopes)
            indices_in_gather = indices_in_gather.view(ctx.num_samples, 1, -1, 1, 1)
            indices_in_gather = indices_in_gather.expand(-1, self.base_leaf.num_channels, -1, -1, num_params)
            indices_in_gather = indices_in_gather.repeat(1, 1, self.base_leaf.cardinality, 1, 1)
            # indices_in_gather: [batch_size, num_channels, num_features, 1]  (last dim is index into num_leaves)
            params = params.gather(dim=-2, index=indices_in_gather)  # -2 is num_leaves dim
            params.squeeze_(-2)  # Remove num_leaves dimension
        assert params.shape == (ctx.num_samples, self.base_leaf.num_channels, self.num_features, num_params)
        return params

    def extra_repr(self):
        return f"num_features={self.num_features}, num_features_out={self.num_features_out}"


class FactorizedLeafSimple(AbstractLayer):
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
        assert (
            num_repetitions == 1
        ), f"FactorizedLeafSimple only supports num_repetitions=1 but was given num_repetitions={num_repetitions}"

        self.base_leaf = base_leaf
        self.num_features_out = num_features_out

        # Size of the factorized groups of RVs
        self.cardinality = int(np.ceil(self.num_features / self.num_features_out))

        # Compute number of dummy nodes that need to be padded
        self.num_dummy_nodes = self.cardinality * self.num_features_out - self.num_features

        # Idea: pad input with "rest" number of dummy nodes
        permutation = torch.randperm(n=self.num_features + self.num_dummy_nodes)
        self.register_buffer("permutation", permutation)

        # Invert permutation
        self.register_buffer("inverse_permutation", torch.argsort(permutation))

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

        # Pad with dummy nodes
        if self.num_dummy_nodes > 0:
            x = torch.cat(
                [x, torch.zeros(x.shape[0], self.num_dummy_nodes, x.shape[-2], x.shape[-1], device=x.device)], dim=1
            )

        # Apply permutation
        x = x[:, self.permutation]

        # Fold into "num_features_out" groups
        x = x.view(x.shape[0], self.num_features_out, self.cardinality, x.shape[-2], x.shape[-1])

        # Sum over the groups
        x = x.sum(dim=2)

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

        # If return_leaf_params is True, we return the parameters of the leaf distribution
        # instead of the samples themselves
        if ctx.return_leaf_params:
            params = self.base_leaf.get_params()
            params = self._index_leaf_params(ctx, indices_out, params=params)
            return params
        else:
            samples = self.base_leaf.sample(ctx)
            samples = self._index_leaf_samples(ctx, indices_out, samples)
            return samples

    def _index_leaf_samples(self, ctx, indices_out, samples):
        # Check that shapes match as expected
        assert samples.shape == (
            ctx.num_samples,
            self.base_leaf.num_channels,
            self.base_leaf.num_features,
            self.base_leaf.num_leaves,
        )
        if ctx.is_differentiable:

            # Unfold into "num_features" + "num_dummy_nodes" by repetition
            indices_out = indices_out.unsqueeze(1)
            indices_out = indices_out.expand(-1, self.cardinality, -1, -1)
            indices_out = indices_out.reshape(
                indices_out.shape[0], self.num_features + self.num_dummy_nodes, indices_out.shape[-1]
            )

            # Invert permutation
            indices_out = indices_out[:, self.inverse_permutation]

            # Remove dummy nodes
            if self.num_dummy_nodes > 0:
                indices_out = indices_out[:, : -self.num_dummy_nodes]

            indices_out = indices_out.unsqueeze(1)  # make space for channel dim
            samples = index_one_hot(samples, index=indices_out, dim=-1)
        else:
            # Unfold into "num_features" + "num_dummy_nodes" by repetition
            indices_out = indices_out.unsqueeze(1).unsqueeze(1)
            indices_out = indices_out.expand(-1, self.base_leaf.num_channels, self.cardinality, -1)
            indices_out = indices_out.reshape(indices_out.shape[0], self.base_leaf.num_channels, self.num_features + self.num_dummy_nodes)

            # Invert permutation
            indices_out = indices_out[:, :, self.inverse_permutation]

            # Remove dummy nodes
            if self.num_dummy_nodes > 0:
                indices_out = indices_out[:, :, : -self.num_dummy_nodes]

            indices_out = indices_out.unsqueeze(-1)
            samples = samples.gather(index=indices_out, dim=-1)
            samples = samples.squeeze(-1)
        return samples

    def _index_leaf_params(self, ctx, indices_out, params):
        """
        Same as _index_leaf_samples, but indexes the parameters of the leaf distribution instead of the samples.
        """
        num_params = params.shape[-1]  # Number of parameters, e.g. 2 for Normal (mu and sigma)
        if ctx.is_differentiable:
            # Unfold into "num_features" + "num_dummy_nodes" by repetition
            indices_out = indices_out.unsqueeze(1)
            indices_out = indices_out.expand(-1, self.cardinality, -1, -1)
            indices_out = indices_out.reshape(
                indices_out.shape[0], self.num_features + self.num_dummy_nodes, indices_out.shape[-1]
            )

            # Invert permutation
            indices_out = indices_out[:, self.inverse_permutation]

            # Remove dummy nodes
            if self.num_dummy_nodes > 0:
                indices_out = indices_out[:, : -self.num_dummy_nodes]

            indices_out = indices_out.unsqueeze(-1)
            params = params.squeeze(-2)  # remove repetition index
            indices_out = indices_out.unsqueeze(-1)  # make space for num_channels dim
            params = index_one_hot(params, index=indices_out, dim=-2)
        else:
            # Unfold into "num_features" + "num_dummy_nodes" by repetition
            indices_out = indices_out.unsqueeze(1).unsqueeze(1)
            indices_out = indices_out.expand(-1, self.base_leaf.num_channels, self.cardinality, -1)
            indices_out = indices_out.reshape(indices_out.shape[0], self.base_leaf.num_channels, self.num_features + self.num_dummy_nodes)

            # Invert permutation
            indices_out = indices_out[:, :, self.inverse_permutation]

            # Remove dummy nodes
            if self.num_dummy_nodes > 0:
                indices_out = indices_out[:, :, : -self.num_dummy_nodes]

            indices_out = indices_out.unsqueeze(-1).unsqueeze(-1)
            indices_out = indices_out.expand(-1, -1, -1, -1, num_params)
            params = params.squeeze(-2)  # remove repetition index
            params = params.expand(ctx.num_samples, -1, -1, -1, -1)
            params = params.gather(index=indices_out, dim=-2)
            params = params.squeeze(-2)  # Remove num_leaves dimension
        assert params.shape == (ctx.num_samples, self.base_leaf.num_channels, self.num_features, num_params)
        return params

    def extra_repr(self):
        return f"num_features={self.num_features}, num_features_out={self.num_features_out}"
