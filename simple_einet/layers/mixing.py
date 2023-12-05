from typing import Tuple

import numpy as np
import torch
from torch import Tensor

from simple_einet.abstract_layers import AbstractSumLayer, logits_to_log_weights
from simple_einet.sampling_utils import SamplingContext, sample_categorical_differentiably, index_one_hot


class MixingLayer(AbstractSumLayer):
    """
    A PyTorch module that implements a linear mixing layer.
    """

    def __init__(
        self,
        num_features: int,
        num_sums_in: int,
        num_sums_out: int,
        dropout: float = 0.0,
    ):
        """
        Initializes a MixingLayer instance.

        Args:
            num_features (int): Number of input and output features.
            num_sums_in (int): Number of input sum nodes.
            num_sums_out (int): Number of output sum nodes.
            dropout (float, optional): Dropout probability. Defaults to 0.0.
        """
        super().__init__(
            num_features, num_repetitions=1, num_sums_in=num_sums_in, num_sums_out=num_sums_out, dropout=dropout
        )

    @property
    def num_features_out(self) -> int:
        return self.num_features

    def weight_shape(self) -> Tuple[int, ...]:
        return self.num_features, self.num_sums_out, self.num_sums_in

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the layer."""
        # Save input if input cache is enabled
        if self._is_input_cache_enabled:
            self._input_cache["in"] = x.clone()

        # Apply dropout: Set random sum node children to 0 (-inf in log domain)
        if self.dropout > 0.0 and self.training:
            dropout_indices = self._bernoulli_dist.sample(x.shape)
            invalid_index = dropout_indices.sum(3) == dropout_indices.shape[3]
            while invalid_index.any():
                # Resample only invalid indices
                dropout_indices[invalid_index] = self._bernoulli_dist.sample(dropout_indices[invalid_index].shape)
                invalid_index = dropout_indices.sum(3) == dropout_indices.shape[3]
            dropout_indices = torch.log(1 - dropout_indices)
            x = x + dropout_indices

        # Get log weights
        log_weights = logits_to_log_weights(self.logits, dim=2).unsqueeze(0)
        lls = torch.logsumexp(x + log_weights, dim=3)

        return lls

    def _sample_from_weights(self, ctx: SamplingContext, log_weights: Tensor):
        if ctx.is_differentiable:
            indices = sample_categorical_differentiably(
                dim=2, is_mpe=ctx.is_mpe, hard=ctx.hard, tau=ctx.tau, log_weights=log_weights
            )
        else:
            if ctx.is_mpe:
                indices = log_weights.argmax(dim=2)
            else:
                # Create categorical distribution to sample from
                dist = torch.distributions.Categorical(logits=log_weights)

                indices = dist.sample()
        return indices

    def _condition_weights_on_evidence(self, ctx: SamplingContext, log_weights: Tensor):
        lls = self._input_cache["in"]

        # Index lls at correct repetitions
        if ctx.is_differentiable:
            p_idxs = ctx.indices_out.view(ctx.num_samples, 1, self.num_sums_out, 1)
            lls = index_one_hot(lls, index=p_idxs, dim=2)
        else:
            p_idxs = ctx.indices_out[..., None, None]
            p_idxs = p_idxs.expand(-1, 1, 1, self.num_sums_in)
            lls = lls.gather(dim=2, index=p_idxs).squeeze(2)

        log_prior = log_weights
        log_posterior = log_prior + lls
        log_posterior = log_posterior - torch.logsumexp(log_posterior, dim=2, keepdim=True)
        log_weights = log_posterior
        return log_weights

    def _select_weights(self, ctx: SamplingContext, logits: Tensor) -> Tensor:
        if ctx.is_differentiable:
            # Index with parent indices
            logits = logits.unsqueeze(0)  # make space for batch dim
            p_idxs = ctx.indices_out.unsqueeze(-1)  # make space for repetition dim
            logits = index_one_hot(logits, index=p_idxs, dim=2)
        else:
            # Index with parent indices
            logits = logits.unsqueeze(0)  # make space for batch dim
            logits = logits.expand(ctx.num_samples, -1, -1, -1)
            p_idxs = ctx.indices_out.unsqueeze(-1).unsqueeze(-1)
            p_idxs = p_idxs.expand(-1, -1, -1, self.num_sums_in)
            logits = logits.gather(dim=2, index=p_idxs)
            # Drop dim which was selected via parent indices
            logits = logits.squeeze(2)

        # Check dimensions
        assert logits.shape == (ctx.num_samples, self.num_features, self.num_sums_in)

        # Project weights into valid space
        log_weights = logits_to_log_weights(logits, dim=-1, temperature=ctx.temperature_sums)
        return log_weights

    def extra_repr(self):
        return "num_features={}, num_sums_in={}, num_sums_out={}".format(
            self.num_features, self.num_sums_in, self.num_sums_out, self.num_repetitions
        )
