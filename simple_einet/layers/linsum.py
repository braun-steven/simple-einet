from typing import Tuple

import numpy as np
import torch

from simple_einet.abstract_layers import AbstractSumLayer, logits_to_log_weights
from simple_einet.layers.einsum import logsumexp
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
        )

        assert self.num_features % LinsumLayer.cardinality == 0, "num_features must be a multiple of cardinality"

        self.out_shape = f"(N, {self.num_features_out}, {self.num_sums_out}, {self.num_repetitions})"

    @property
    def num_features_out(self) -> int:
        return self.num_features // LinsumLayer.cardinality

    def weight_shape(self) -> Tuple[int, ...]:
        return self.num_features_out, self.num_sums_in, self.num_sums_out, self.num_repetitions


    def forward_tdi(self, log_exp_ch: torch.Tensor, log_var_ch: torch.Tensor, dropout_inference=None):
        """
        Einsum layer dropout inference pass.

        Args:
            log_exp_ch: Input expectations.
            log_var_ch: Input variances.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output expectations and variances
        """

        # Dimensions
        N, D, C, R = log_exp_ch.size()
        D_out = D // 2

        #################
        # PRODUCT LAYER #
        #################

        # ---------------------------------------------------
        # | 1. Product expectation (default log-likelihood) |
        # ---------------------------------------------------

        # Get left and right partition probs
        log_exp_left = log_exp_ch[:, 0::2]
        log_exp_right = log_exp_ch[:, 1::2]
        log_exp_prod = (log_exp_left + log_exp_right).unsqueeze(3)  # N x D/2 x Sin x 1 x R

        # -----------------------
        # | 2. Product variance |
        # -----------------------

        # Get left and right partition vars
        log_var_left = log_var_ch[:, 0::2]
        log_var_right = log_var_ch[:, 1::2]

        log_exp_sq_left = log_exp_left * 2
        log_exp_sq_right = log_exp_right * 2

        log_var_right_term = log_exp_sq_left + log_exp_sq_right

        log_var_left_term_left = logsumexp((log_var_left, log_exp_sq_left))
        log_var_left_term_right = logsumexp((log_var_right, log_exp_sq_right))

        log_var_left_term = log_var_left_term_left + log_var_left_term_right

        log_var_prod = logsumexp((log_var_left_term, log_var_right_term), mask=[1, -1])

        #############
        # SUM LAYER #
        #############

        # Prepare constants
        # If dropout at inference time is set, use this instead
        if dropout_inference is not None:
            log_q = np.log(1 - dropout_inference)
            log_p = np.log(dropout_inference)
        else:
            log_q = torch.log(1 - self.dropout)
            log_p = torch.log(self.dropout)

        # Get log weights
        log_weights = logits_to_log_weights(self.logits, dim=1).unsqueeze(0)

        # ----------------------
        # | 3. Sum expectation |
        # ----------------------

        log_exp_sum = log_q + torch.logsumexp(log_exp_prod + log_weights, dim=2)  # N x D/2 x Sout x R

        # -------------------
        # | 4. Sum variance |
        # -------------------

        log_weights_sq = log_weights * 2
        log_exp_prod_sq = log_exp_prod * 2
        log_var_prod = log_var_prod.unsqueeze(3)

        log_var_plus_exp = torch.logsumexp(torch.stack((log_var_prod, log_exp_prod_sq + log_p), dim=-1), dim=-1)
        log_var_sum = log_q + torch.logsumexp(log_weights_sq + log_var_plus_exp, dim=2)

        return log_exp_sum, log_var_sum

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
        log_prior = log_weights
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
