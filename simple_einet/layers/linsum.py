from typing import Tuple

import torch

from simple_einet.abstract_layers import AbstractSumLayer, logits_to_log_weights
from simple_einet.sampling_utils import (
    index_one_hot,
    sample_categorical_differentiably,
)


class LinsumLayer(AbstractSumLayer):
    """
    Similar to Einsum but with a linear combination of the input channels for each output channel compared to
    the cross-product combination that is applied in an EinsumLayer.
    """

    cardinality = 2  # Cardinality of the layer

    def __init__(
        self,
        num_features: int,
        num_sums_in: int,
        num_sums_out: int,
        num_repetitions: int = 1,
        dropout: float = 0.0,
        **kwargs,
    ):
        """
        Initializes a LinsumLayer instance.

        Args:
            num_features (int): The number of input features.
            num_sums_in (int): The number of input sums.
            num_sums_out (int): The number of output sums.
            num_repetitions (int, optional): The number of times to repeat the layer. Defaults to 1.
            dropout (float, optional): The dropout probability. Defaults to 0.0.
        """

        # Number of features to be padded (assign this before the super().__init__ call, so it can be used in the
        # super initializer)
        self._pad = num_features % LinsumLayer.cardinality

        super().__init__(
            num_features=num_features,
            num_sums_in=num_sums_in,
            num_sums_out=num_sums_out,
            num_repetitions=num_repetitions,
            dropout=dropout,
            **kwargs,
        )

        # assert self.num_features % LinsumLayer.cardinality == 0, "num_features must be a multiple of cardinality"

        self.out_shape = f"(N, {self.num_features_out}, {self.num_sums_out}, {self.num_repetitions})"

    @property
    def num_features_out(self) -> int:
        return (self.num_features + self._pad) // LinsumLayer.cardinality

    def weight_shape(self) -> Tuple[int, ...]:
        return self.num_features_out, self.num_sums_in, self.num_sums_out, self.num_repetitions

    def forward(self, x: torch.Tensor):
        """
        Einsum layer forward pass.

        Args:
            x: Input of shape [batch, in_features, num_sums_in, num_repetitions].

        Returns:
            torch.Tensor: Output of shape [batch, ceil(in_features/2), channel * channel].
        """

        # Dimensions
        N, D, C, R = x.size()
        D_out = D // 2

        # Get left and right partition probs
        left = x[:, 0::2]
        right = x[:, 1::2]


        if self._pad > 0:
            # Add dummy marginalized RVs
            right = torch.cat([right, torch.zeros_like(right[:, : self._pad])], dim=1)

        prod_output = (left + right).unsqueeze(3)  # N x D/2 x Sin x 1 x R

        # Apply dropout: Set random sum node children to 0 (-inf in log domain)
        if self.dropout > 0.0 and self.training:
            dropout_indices = self._bernoulli_dist.sample(prod_output.shape)
            invalid_index = dropout_indices.sum(2) == dropout_indices.shape[2]
            while invalid_index.any():
                # Resample only invalid indices
                dropout_indices[invalid_index] = self._bernoulli_dist.sample(dropout_indices[invalid_index].shape)
                invalid_index = dropout_indices.sum(2) == dropout_indices.shape[2]
            dropout_indices = torch.log(1 - dropout_indices)
            prod_output = prod_output + dropout_indices

        # Get log weights
        log_weights = logits_to_log_weights(self.logits, dim=1).unsqueeze(0)
        prob = torch.logsumexp(prod_output + log_weights, dim=2)  # N x D/2 x Sout x R

        # Save input if input cache is enabled
        if self._is_input_cache_enabled:
            self._input_cache["left"] = left
            self._input_cache["right"] = right

        return prob

    def _sample_from_weights(self, ctx, log_weights):
        if ctx.is_differentiable:  # Differentiable sampling
            indices = sample_categorical_differentiably(
                dim=-1, is_mpe=ctx.is_mpe, hard=ctx.hard, tau=ctx.tau, log_weights=log_weights
            )
            indices = indices.repeat_interleave(2, dim=1)
            indices = indices.view(ctx.num_samples, -1, self.num_sums_in)

        else:  # Non-differentiable sampling
            if ctx.is_mpe:
                indices = log_weights.argmax(dim=2)
            else:
                # Create categorical distribution to sample from
                dist = torch.distributions.Categorical(logits=log_weights)
                indices = dist.sample()

            indices = indices.repeat_interleave(2, dim=1)
            indices = indices.view(ctx.num_samples, -1)


        if self._pad > 0:
            # Cut off dummy marginalized RVs
            indices = indices[:, : -self._pad]

        return indices

    def _condition_weights_on_evidence(self, ctx, log_weights):
        # Extract input cache
        input_cache_left = self._input_cache["left"]
        input_cache_right = self._input_cache["right"]

        # Index repetition
        if ctx.is_differentiable:
            r_idxs = ctx.indices_repetition.view(ctx.num_samples, 1, 1, self.num_repetitions)
            lls_left = index_one_hot(input_cache_left, index=r_idxs, dim=-1)
            lls_right = index_one_hot(input_cache_right, index=r_idxs, dim=-1)
        else:
            r_idxs = ctx.indices_repetition[..., None, None, None]
            r_idxs = r_idxs.expand(-1, self.num_features_out, self.num_sums_in, -1)
            lls_left = input_cache_left.gather(index=r_idxs, dim=-1).squeeze(-1)
            lls_right = input_cache_right.gather(index=r_idxs, dim=-1).squeeze(-1)
        lls = (lls_left + lls_right).view(ctx.num_samples, self.num_features_out, self.num_sums_in)
        log_prior = log_weights  # Shape: [batch, num_features_out, num_sums_in]
        log_posterior = log_prior + lls
        log_posterior = log_posterior - torch.logsumexp(log_posterior, dim=2, keepdim=True)
        log_weights = log_posterior
        return log_weights

    def _select_weights(self, ctx, logits):
        if ctx.is_differentiable:
            # Index sums_out
            logits = logits.unsqueeze(0)  # make space for batch dim
            p_idxs = ctx.indices_out.unsqueeze(2).unsqueeze(-1)

            # Index into the "num_sums_out" dimension
            logits = index_one_hot(logits, index=p_idxs, dim=3)
            assert logits.shape == (
                ctx.num_samples,
                self.num_features_out,
                self.num_sums_in,
                self.num_repetitions,
            )

            # Index repetition
            r_idxs = ctx.indices_repetition.view(ctx.num_samples, 1, 1, self.num_repetitions)
            logits = index_one_hot(logits, index=r_idxs, dim=3)

        else:
            # Index sums_out
            logits = logits.unsqueeze(0)  # make space for batch dim
            logits = logits.expand(ctx.num_samples, -1, -1, -1, -1)
            p_idxs = ctx.indices_out[..., None, None, None]  # make space for repetition dim
            p_idxs = p_idxs.expand(-1, -1, self.num_sums_in, -1, self.num_repetitions)

            logits = logits.gather(dim=3, index=p_idxs)  # index out_channels
            logits = logits.squeeze(3)  # squeeze out_channels dimension (is 1 at this point)

            # Index repetitions
            r_idxs = ctx.indices_repetition[..., None, None, None]
            r_idxs = r_idxs.expand(-1, self.num_features_out, self.num_sums_in, -1)
            logits = logits.gather(dim=3, index=r_idxs)
            logits = logits.squeeze(3)
        # Check dimensions
        assert logits.shape == (ctx.num_samples, self.num_features_out, self.num_sums_in)

        # Project logits to log weights
        log_weights = logits_to_log_weights(logits, dim=2, temperature=ctx.temperature_sums)
        return log_weights

    def extra_repr(self):
        return (
            "num_features={}, num_sums_in={}, num_sums_out={}, num_repetitions={}, out_shape={}, "
            "weight_shape={}".format(
                self.num_features,
                self.num_sums_in,
                self.num_sums_out,
                self.num_repetitions,
                self.out_shape,
                self.weight_shape(),
            )
        )


class LinsumLayer2(AbstractSumLayer):
    """
    Similar to Einsum but with a linear combination of the input channels for each output channel compared to
    the cross-product combination that is applied in an EinsumLayer.
    """

    cardinality = 2  # Cardinality of the layer

    def __init__(
        self,
        num_features: int,
        num_sums_in: int,
        num_sums_out: int,
        num_repetitions: int = 1,
        dropout: float = 0.0,
        **kwargs,
    ):
        """
        Initializes a LinsumLayer instance.

        Args:
            num_features (int): The number of input features.
            num_sums_in (int): The number of input sums.
            num_sums_out (int): The number of output sums.
            num_repetitions (int, optional): The number of times to repeat the layer. Defaults to 1.
            dropout (float, optional): The dropout probability. Defaults to 0.0.
        """
        super().__init__(
            num_features=num_features,
            num_sums_in=num_sums_in,
            num_sums_out=num_sums_out,
            num_repetitions=num_repetitions,
            dropout=dropout,
            **kwargs,
        )

        self._pad = self.num_features % LinsumLayer2.cardinality
        self.out_shape = f"(N, {self.num_features_out}, {self.num_sums_out}, {self.num_repetitions})"

    @property
    def num_features_out(self) -> int:
        return (self.num_features + self._pad) // LinsumLayer2.cardinality

    def weight_shape(self) -> Tuple[int, ...]:
        return self.num_features, self.num_sums_in, self.num_sums_out, self.num_repetitions

    def forward(self, x: torch.Tensor):
        """
        Einsum layer forward pass.

        Args:
            x: Input of shape [batch, in_features, num_sums_in, num_repetitions].

        Returns:
            torch.Tensor: Output of shape [batch, ceil(in_features/2), channel * channel].
        """
        # Save input if input cache is enabled
        if self._is_input_cache_enabled:
            self._input_cache["x"] = x

        # Get log weights
        log_weights = logits_to_log_weights(self.logits, dim=1).unsqueeze(0)
        x = x.unsqueeze(3)
        sum_output = torch.logsumexp(x + log_weights, dim=2)  # N x D x Sout x R

        # Get left and right partition probs
        left = sum_output[:, 0::2]
        right = sum_output[:, 1::2]

        if self._pad > 0:
            # Add dummy marginalized RVs
            right = torch.cat([right, torch.zeros_like(right[:, : self._pad])], dim=1)

        prod_output = left + right  # N x D/2 x Sout x 1 x R

        # Apply dropout: Set random sum node children to 0 (-inf in log domain)
        if self.dropout > 0.0 and self.training:
            dropout_indices = self._bernoulli_dist.sample(prod_output.shape)
            invalid_index = dropout_indices.sum(2) == dropout_indices.shape[2]
            while invalid_index.any():
                # Resample only invalid indices
                dropout_indices[invalid_index] = self._bernoulli_dist.sample(dropout_indices[invalid_index].shape)
                invalid_index = dropout_indices.sum(2) == dropout_indices.shape[2]
            dropout_indices = torch.log(1 - dropout_indices)
            prod_output = prod_output + dropout_indices

        return prod_output

    def _sample_from_weights(self, ctx, log_weights):
        if ctx.is_differentiable:  # Differentiable sampling
            indices = sample_categorical_differentiably(
                dim=-1, is_mpe=ctx.is_mpe, hard=ctx.hard, tau=ctx.tau, log_weights=log_weights
            )
            indices = indices.view(ctx.num_samples, -1, self.num_sums_in)

        else:  # Non-differentiable sampling
            if ctx.is_mpe:
                indices = log_weights.argmax(dim=2)
            else:
                # Create categorical distribution to sample from
                dist = torch.distributions.Categorical(logits=log_weights)
                indices = dist.sample()

            indices = indices.view(ctx.num_samples, -1)


        if self._pad > 0:
            # Cut off dummy marginalized RVs
            indices = indices[:, : -self._pad]

        return indices

    def _condition_weights_on_evidence(self, ctx, log_weights):
        # Extract input cache
        input_cache = self._input_cache["x"]

        # Index repetition
        if ctx.is_differentiable:
            r_idxs = ctx.indices_repetition.view(ctx.num_samples, 1, 1, self.num_repetitions)
            lls = index_one_hot(input_cache, index=r_idxs, dim=-1)
        else:
            r_idxs = ctx.indices_repetition[..., None, None, None]
            r_idxs = r_idxs.expand(-1, self.num_features, self.num_sums_in, -1)
            lls = input_cache.gather(index=r_idxs, dim=-1).squeeze(-1)
        log_prior = log_weights  # Shape: [batch, num_features_out, num_sums_in]
        log_posterior = log_prior + lls
        log_posterior = log_posterior - torch.logsumexp(log_posterior, dim=2, keepdim=True)
        log_weights = log_posterior
        return log_weights

    def _select_weights(self, ctx, logits):
        if ctx.is_differentiable:
            # Index sums_out
            logits = logits.unsqueeze(0)  # make space for batch dim
            p_idxs = ctx.indices_out.repeat_interleave(2, dim=1)

            if self._pad > 0:
                # Cut off dummy marginalized RVs
                p_idxs = p_idxs[:, : -self._pad]

            p_idxs = p_idxs.unsqueeze(2).unsqueeze(-1)

            # Index into the "num_sums_out" dimension
            logits = index_one_hot(logits, index=p_idxs, dim=3)
            assert logits.shape == (
                ctx.num_samples,
                self.num_features,
                self.num_sums_in,
                self.num_repetitions,
            )

            # Index repetition
            r_idxs = ctx.indices_repetition.view(ctx.num_samples, 1, 1, self.num_repetitions)
            logits = index_one_hot(logits, index=r_idxs, dim=3)

        else:
            # Index sums_out
            logits = logits.unsqueeze(0)  # make space for batch dim
            logits = logits.expand(ctx.num_samples, -1, -1, -1, -1)

            p_idxs = ctx.indices_out.repeat_interleave(2, dim=1)

            if self._pad > 0:
                # Cut off dummy marginalized RVs
                p_idxs = p_idxs[:, : -self._pad]

            p_idxs = p_idxs[..., None, None, None]  # make space for repetition dim
            p_idxs = p_idxs.expand(-1, -1, self.num_sums_in, -1, self.num_repetitions)
            logits = logits.gather(dim=3, index=p_idxs)  # index out_channels
            logits = logits.squeeze(3)  # squeeze out_channels dimension (is 1 at this point)

            # Index repetitions
            r_idxs = ctx.indices_repetition[..., None, None, None]
            r_idxs = r_idxs.expand(-1, self.num_features, self.num_sums_in, -1)
            logits = logits.gather(dim=3, index=r_idxs)
            logits = logits.squeeze(3)
        # Check dimensions
        assert logits.shape == (ctx.num_samples, self.num_features, self.num_sums_in)

        # Project logits to log weights
        log_weights = logits_to_log_weights(logits, dim=2, temperature=ctx.temperature_sums)
        return log_weights

    def extra_repr(self):
        return (
            "num_features={}, num_sums_in={}, num_sums_out={}, num_repetitions={}, out_shape={}, "
            "weight_shape={}".format(
                self.num_features,
                self.num_sums_in,
                self.num_sums_out,
                self.num_repetitions,
                self.out_shape,
                self.weight_shape(),
            )
        )
