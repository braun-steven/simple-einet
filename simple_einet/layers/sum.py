from typing import Tuple

import numpy as np
import torch

from simple_einet.abstract_layers import AbstractSumLayer, logits_to_log_weights
from simple_einet.sampling_utils import SamplingContext, index_one_hot, sample_categorical_differentiably


class SumLayer(AbstractSumLayer):
    """
    Sum Node Layer that sums over all children in a scope set.

    Attributes:
        num_sums_in (int): Number of input sum nodes.
        num_sums_out (int): Multiplicity of a sum node for a given scope set.
        num_features (int): Number of input features.
        num_repetitions (int): Number of layer repetitions in parallel.
        dropout (torch.nn.Parameter): Dropout percentage.
        out_shape (str): Output shape of the layer.
    """

    def __init__(
        self,
        num_sums_in: int,
        num_features: int,
        num_sums_out: int,
        num_repetitions: int = 1,
        dropout: float = 0.0,
    ):
        """
        Create a Sum layer.

        Input is expected to be of shape [n, d, ic, r].
        Output will be of shape [n, d, oc, r].

        Args:
            num_sums_in (int): Number of input sum nodes.
            num_features (int): Number of input features.
            num_sums_out (int): Multiplicity of a sum node for a given scope set.
            num_repetitions (int, optional): Number of layer repetitions in parallel. Defaults to 1.
            dropout (float, optional): Dropout percentage. Defaults to 0.0.
        """
        super().__init__(
            num_sums_in=num_sums_in,
            num_features=num_features,
            num_sums_out=num_sums_out,
            num_repetitions=num_repetitions,
            dropout=dropout,
        )

        # Weights, such that each sumnode has its own weights
        self.out_shape = f"(N, {self.num_features}, {self.num_sums_out}, {self.num_repetitions})"

    @property
    def num_features_out(self) -> int:
        return self.num_features

    def weight_shape(self) -> Tuple[int, ...]:
        return self.num_features_out, self.num_sums_in, self.num_sums_out, self.num_repetitions

    def forward(self, x: torch.Tensor):
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
        log_weights = logits_to_log_weights(self.logits, dim=1).unsqueeze(0)
        x = x.unsqueeze(3)  # Make space for num_sums_out dim
        lls = torch.logsumexp(x + log_weights, dim=2)

        # Assert correct dimensions
        assert lls.size() == (x.shape[0], x.shape[1], self.num_sums_out, self.num_repetitions)

        return lls

    def _select_weights(self, ctx: SamplingContext, logits: torch.Tensor) -> torch.Tensor:
        if ctx.is_differentiable:
            # Index with parent indices
            logits = logits.unsqueeze(0)  # make space for batch dim
            p_idxs = ctx.indices_out.unsqueeze(2).unsqueeze(-1)
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
        assert logits.shape == (ctx.num_samples, self.num_features, self.num_sums_in)

        # Project logits to log weights
        log_weights = logits_to_log_weights(logits, dim=2, temperature=ctx.temperature_sums)
        return log_weights

    def _sample_from_weights(self, ctx: SamplingContext, log_weights: torch.Tensor) -> torch.Tensor:
        if ctx.is_differentiable:  # Differentiable sampling
            indices = sample_categorical_differentiably(
                dim=-1, is_mpe=ctx.is_mpe, hard=ctx.hard, tau=ctx.tau, log_weights=log_weights
            )
        else:  # Non-differentiable sampling
            if ctx.is_mpe:
                indices = log_weights.argmax(dim=2)
            else:
                # Create categorical distribution to sample from
                dist = torch.distributions.Categorical(logits=log_weights)
                indices = dist.sample()
        return indices

    def _condition_weights_on_evidence(self, ctx: SamplingContext, log_weights: torch.Tensor):
        lls = self._input_cache["in"]

        # Index repetition
        if ctx.is_differentiable:
            r_idxs = ctx.indices_repetition.view(ctx.num_samples, 1, 1, self.num_repetitions)
            lls = index_one_hot(lls, index=r_idxs, dim=3)
        else:
            r_idxs = ctx.indices_repetition[..., None, None, None]
            r_idxs = r_idxs.expand(-1, self.num_features, self.num_sums_in, 1)
            lls = lls.gather(dim=3, index=r_idxs).squeeze(3)

        log_prior = log_weights
        log_posterior = log_prior + lls
        log_posterior = log_posterior - torch.logsumexp(log_posterior, dim=2, keepdim=True)
        log_weights = log_posterior
        return log_weights

    def extra_repr(self):
        return "num_features={}, num_sums_in={}, num_sums_out={}, num_repetitions={}, dropout={}, out_shape={}".format(
            self.num_features,
            self.num_sums_in,
            self.num_sums_out,
            self.num_repetitions,
            self.dropout.item(),
            self.out_shape,
        )
