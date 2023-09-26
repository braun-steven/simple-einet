from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from icecream import ic

from simple_einet.layers import AbstractLayer
from simple_einet.type_checks import check_valid
from simple_einet.sampling_utils import SamplingContext, sample_categorical_differentiably, index_one_hot


def logsumexp(tensors, mask=None, dim=-1):
    """
    Source: https://github.com/pytorch/pytorch/issues/32097

    Logsumexp with custom scalar mask to allow for negative values in the sum.

    Args:
        tensors (torch.Tensor, List[torch.Tensor]): The tensors to sum.
        mask (torch.Tensor, optional): The mask to apply to the sum. Defaults to None.
        dim (int, optional): The dimension to sum over. Defaults to -1.

    Returns:
        torch.Tensor: The summed tensor.
    """
    # Ensure that everything is a tensor
    if type(tensors) == list or type(tensors) == tuple:
        tensors = torch.stack(tensors, dim=dim)
    if mask is None:
        mask = torch.ones(tensors.shape[dim], device=tensors.device)
    else:
        if type(mask) == list or type(mask) == tuple:
            mask = torch.tensor(mask, device=tensors.device)
        assert mask.shape == (tensors.shape[dim],), "Invalid mask shape"

    maxes = torch.max(tensors, dim=dim)[0]
    return ((tensors - maxes.unsqueeze(dim)).exp() * mask).sum(dim=dim).log() + maxes


class LinsumLayer(AbstractLayer):
    """
    Similar to Einsum but with a linear combination of the input channels for each output channel compared to
    the cross-product combination that is applied in an EinsumLayer.

    Attributes:
        num_sums_in (int): The number of input sums.
        num_sums_out (int): The number of output sums.
        num_repetitions (int): The number of repetitions of the layer.
        weights (torch.Tensor): The weights of the layer.
        dropout (float): The dropout probability.
        _bernoulli_dist (torch.distributions.Bernoulli): The Bernoulli distribution used for dropout.
        _is_input_cache_enabled (bool): Whether the input cache is enabled.
        _input_cache_left (torch.Tensor): The left input cache.
        _input_cache_right (torch.Tensor): The right input cache.
        cardinality (int): The cardinality of the layer.
    """

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
        super().__init__(num_features, num_repetitions)

        self.num_sums_in = check_valid(num_sums_in, int, 1)
        self.num_sums_out = check_valid(num_sums_out, int, 1)
        cardinality = 2  # Fixed to binary graphs for now
        self.cardinality = check_valid(cardinality, int, 2, num_features + 1)
        self.num_features_out = np.ceil(self.num_features / self.cardinality).astype(int)

        ws = self._init_weights()

        self.weights = nn.Parameter(ws)

        # Dropout
        self.dropout = check_valid(dropout, expected_type=float, lower_bound=0.0, upper_bound=1.0)
        self.dropout = nn.Parameter(torch.tensor(self.dropout), requires_grad=False)
        self._bernoulli_dist = torch.distributions.Bernoulli(probs=self.dropout)

        # Necessary for sampling with evidence: Save input during forward pass.
        self._is_input_cache_enabled = False
        self._input_cache_left = None
        self._input_cache_right = None

        self.out_shape = f"(N, {self.num_features_out}, {self.num_sums_out}, {self.num_repetitions})"

    def _init_weights(self):
        """
        Initializes the weights of the layer.
        """
        # Weights, such that each sumnode has its own weights
        ws = torch.randn(
            self.num_features // self.cardinality,
            self.num_sums_in,
            self.num_sums_out,
            self.num_repetitions,
        )
        return ws

    def _get_normalized_log_weights(self):
        return F.log_softmax(self.weights, dim=1)

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
        log_weights = self._get_normalized_log_weights().unsqueeze(0)

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
            x: Input of shape [batch, in_features, channel].

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
        log_weights = self._get_normalized_log_weights().unsqueeze(0)
        prob = torch.logsumexp(prod_output + log_weights, dim=2)  # N x D/2 x Sout x R

        # Save input if input cache is enabled
        if self._is_input_cache_enabled:
            self._input_cache_left = left
            self._input_cache_right = right

        return prob

    def sample(self, num_samples: int, context: SamplingContext) -> SamplingContext:
        """
        Samples from the weights of the EinsumLayer and returns a SamplingContext object
        containing the sampled indices.

        Args:
            num_samples (int): The number of samples to generate.
            context (SamplingContext): The SamplingContext object containing the indices
                used for sampling.

        Returns:
            SamplingContext: The SamplingContext object containing the sampled indices.
        """
        # Sum weights are of shape: [D, IC//2, IC//2, OC, R]
        # We now want to use `indices` to access one in_channel for each in_feature x
        # out_channels block index is of size in_feature
        weights = self.weights
        (
            out_features,
            in_channels,
            out_channels,
            num_repetitions,
        ) = weights.shape
        num_samples = context.num_samples

        self._check_indices_repetition(context)

        if not context.is_differentiable:
            # Index sums_out
            weights = weights.unsqueeze(0)  # make space for batch dim
            weights = weights.expand(num_samples, -1, -1, -1, -1)
            p_idxs = context.indices_out[..., None, None, None]  # make space for repetition dim
            p_idxs = p_idxs.expand(-1, -1, in_channels, -1, num_repetitions)
            weights = weights.gather(dim=3, index=p_idxs)  # index out_channels
            weights = weights.squeeze(3)  # squeeze out_channels dimension (is 1 at this point)

            # Index repetitions
            r_idxs = context.indices_repetition[..., None, None, None]
            r_idxs = r_idxs.expand(-1, out_features, in_channels, -1)
            weights = weights.gather(dim=3, index=r_idxs)
            weights = weights.squeeze(3)

        else:
            # TODO: implement
            raise NotImplementedError()
            # Index sums_out
            weights = weights.unsqueeze(0)  # make space for batch dim
            p_idxs = context.indices_out[..., None, None, None]  # make space for repetition dim
            weights = index_one_hot(weights, index=p_idxs, dim=2)
            assert weights.shape == (
                num_samples,
                out_features,
                num_repetitions,
                in_channels,
                in_channels,
            )

            # Index repetition
            r_idxs = context.indices_repetition.view(num_samples, 1, num_repetitions, 1, 1)
            weights = index_one_hot(weights, index=r_idxs, dim=2)

        # Check dimensions
        assert weights.shape == (num_samples, out_features, in_channels)

        # Apply softmax to ensure they are proper probabilities
        log_weights = F.log_softmax(weights * context.temperature_sums, dim=2)

        # If evidence is given, adjust the weights with the likelihoods of the observed paths
        if self._is_input_cache_enabled and self._input_cache_left is not None:
            # TODO: Implement
            raise NotImplementedError()
            # Index repetition
            if context.is_differentiable:
                r_idxs = context.indices_repetition.view(num_samples, 1, 1, num_repetitions)
                lls_left = index_one_hot(self._input_cache_left, index=r_idxs, dim=-1).unsqueeze(3)
                lls_right = index_one_hot(self._input_cache_right, index=r_idxs, dim=-1).unsqueeze(2)
            else:
                r_idxs = context.indices_repetition[..., None, None, None]
                r_idxs = r_idxs.expand(-1, out_features, in_channels, -1)
                lls_left = self._input_cache_left.gather(index=r_idxs, dim=-1).squeeze(-1).unsqueeze(3)
                lls_right = self._input_cache_right.gather(index=r_idxs, dim=-1).squeeze(-1).unsqueeze(2)

            lls = (lls_left + lls_right).view(num_samples, out_features, in_channels**2)
            log_prior = log_weights
            log_posterior = log_prior + lls
            log_posterior = log_posterior - torch.logsumexp(log_posterior, dim=2, keepdim=True)
            log_weights = log_posterior

        # Sample/mpe from the logweights
        if not context.is_differentiable:  # Non-differentiable sampling
            if context.is_mpe:
                indices = log_weights.argmax(dim=2)
            else:
                # Create categorical distribution to sample from
                dist = torch.distributions.Categorical(logits=log_weights)
                indices = dist.sample()

            indices = indices.repeat_interleave(2, dim=1)
            indices = indices.view(num_samples, -1)

        else:  # Differentiable sampling
            raise NotImplementedError()
            indices = sample_categorical_differentiably(
                log_weights,
                dim=-1,
                is_mpe=context.is_mpe,
                hard=context.hard,
                tau=context.tau,
            )
            indices = indices.unsqueeze(-1)
            indices_a = (indices * self.unraveled_channel_indices_oh_0).sum(-2)
            indices_b = (indices * self.unraveled_channel_indices_oh_1).sum(-2)
            indices = torch.stack((indices_a, indices_b), dim=2)
            indices = indices.view(num_samples, -1, in_channels)

        context.indices_out = indices
        return context

    def _check_indices_repetition(self, context: SamplingContext):
        assert context.indices_repetition.shape[0] == context.indices_out.shape[0]
        if self.num_repetitions > 1 and context.indices_repetition is None:
            raise Exception(
                f"Sum layer has multiple repetitions (num_repetitions=={self.num_repetitions}) "
                f"but indices_repetition argument was None, expected a Long tensor size #samples."
            )
        if self.num_repetitions == 1 and context.indices_repetition is None:
            context.indices_repetition = torch.zeros(context.num_samples, dtype=torch.int, device=self.__device)

    def _enable_input_cache(self):
        """Enables the input cache. This will store the input in forward passes into
        `self.__input_cache`."""
        self._is_input_cache_enabled = True

    def _disable_input_cache(self):
        """Disables and clears the input cache."""
        self._is_input_cache_enabled = False
        self._input_cache_left = None
        self._input_cache_right = None

    def extra_repr(self):
        return "num_features={}, num_sums_in={}, num_sums_out={}, out_shape={}, " "weights_shape={}".format(
            self.num_features,
            self.num_sums_in,
            self.num_sums_out,
            self.out_shape,
            self.weights.shape,
        )


class LinsumLayerLogWeights(LinsumLayer):
    def _init_weights(self):
        # Weights, such that each sumnode has its own weights
        log_weights = torch.rand(
            self.num_features // self.cardinality,
            self.num_sums_in,
            self.num_sums_out,
            self.num_repetitions,
        ).log()
        return log_weights

    def _get_normalized_log_weights(self):
        log_weights = self.weights - self.weights.logsumexp(dim=1, keepdim=True)
        return log_weights


class EinsumLayer(AbstractLayer):
    def __init__(
        self,
        num_features: int,
        num_sums_in: int,
        num_sums_out: int,
        num_repetitions: int = 1,
        dropout: float = 0.0,
    ):
        """
        EinsumLayer is a PyTorch module that implements the Einsum layer for the Einet model.

        Args:
            num_features (int): The number of input features.
            num_sums_in (int): The number of input sum nodes.
            num_sums_out (int): The number of output sum nodes.
            num_repetitions (int, optional): The number of repetitions. Defaults to 1.
            dropout (float, optional): The dropout probability. Defaults to 0.0.
        """
        super().__init__(num_features, num_repetitions)

        self.num_sums_in = check_valid(num_sums_in, int, 1)
        self.num_sums_out = check_valid(num_sums_out, int, 1)
        cardinality = 2  # Fixed to binary graphs for now
        self.cardinality = check_valid(cardinality, int, 2, num_features + 1)
        self.num_features_out = np.ceil(self.num_features / self.cardinality).astype(int)

        # Weights, such that each sumnode has its own weights
        ws = torch.randn(
            self.num_features // cardinality,
            self.num_sums_out,
            self.num_repetitions,
            self.num_sums_in,
            self.num_sums_in,
        )

        self.weights = nn.Parameter(ws)

        # Create index map from flattened to coordinates (only needed in sampling)
        self.register_buffer(
            "unraveled_channel_indices",
            torch.tensor([(i, j) for i in range(self.num_sums_in) for j in range(self.num_sums_in)]),
        )

        # Create index map from flattened to coordinates (only needed in differentiable sampling)
        self.register_buffer(
            "unraveled_channel_indices_oh_0",
            torch.nn.functional.one_hot(torch.arange(self.num_sums_in).repeat_interleave(self.num_sums_in))
            .unsqueeze(0)
            .unsqueeze(0),
        )

        self.register_buffer(
            "unraveled_channel_indices_oh_1",
            torch.nn.functional.one_hot(torch.arange(self.num_sums_in).repeat(self.num_sums_in))
            .unsqueeze(0)
            .unsqueeze(0),
        )

        # Dropout
        self.dropout = check_valid(dropout, expected_type=float, lower_bound=0.0, upper_bound=1.0)
        self._bernoulli_dist = torch.distributions.Bernoulli(probs=self.dropout)

        # Necessary for sampling with evidence: Save input during forward pass.
        self._is_input_cache_enabled = False
        self._input_cache_left = None
        self._input_cache_right = None

        self.out_shape = f"(N, {self.num_features_out}, {self.num_sums_out}, {self.num_repetitions})"

    def forward(self, x: torch.Tensor):
        """
        Einsum layer forward pass.

        Args:
            x: Input of shape [batch, in_features, channel].

        Returns:
            torch.Tensor: Output of shape [batch, ceil(in_features/2), channel * channel].
        """

        # Apply dropout: Set random sum node children to 0 (-inf in log domain)
        if self.dropout > 0.0 and self.training:
            dropout_indices = self._bernoulli_dist.sample(x.shape).bool()
            x[dropout_indices] = np.NINF

        # Dimensions
        N, D, C, R = x.size()
        D_out = D // 2

        # Get left and right partition probs
        left = x[:, 0::2]
        right = x[:, 1::2]

        # Prepare for LogEinsumExp trick (see paper for details)
        left_max = torch.max(left, dim=2, keepdim=True)[0]
        left_prob = torch.exp(left - left_max)
        right_max = torch.max(right, dim=2, keepdim=True)[0]
        right_prob = torch.exp(right - right_max)

        # Project weights into valid space
        weights = self.weights
        weights = weights.view(D_out, self.num_sums_out, self.num_repetitions, -1)
        weights = F.softmax(weights, dim=-1)
        weights = weights.view(self.weights.shape)

        # Einsum operation for sum(product(x))
        # n: batch, i: left-channels, j: right-channels, d:features, o: output-channels, r: repetitions
        prob = torch.einsum("ndir,ndjr,dorij->ndor", left_prob, right_prob, weights)

        # LogEinsumExp trick, re-add the max
        prob = torch.log(prob) + left_max + right_max

        # Save input if input cache is enabled
        if self._is_input_cache_enabled:
            self._input_cache_left = left
            self._input_cache_right = right

        return prob

    def sample(self, num_samples: int, context: SamplingContext) -> SamplingContext:
        """
        Samples from the weights of the EinsumLayer using the provided SamplingContext.

        Args:
            num_samples (int): The number of samples to generate.
            context (SamplingContext): The SamplingContext to use for sampling.

        Returns:
            SamplingContext: The updated SamplingContext.
        """
        # Sum weights are of shape: [D, IC//2, IC//2, OC, R]
        # We now want to use `indices` to access one in_channel for each in_feature x
        # out_channels block index is of size in_feature
        weights = self.weights
        (
            out_features,
            out_channels,
            num_repetitions,
            in_channels,
            in_channels,
        ) = weights.shape
        num_samples = context.num_samples

        self._check_indices_repetition(context)

        if not context.is_differentiable:
            # Index sums_out
            weights = weights.unsqueeze(0)  # make space for batch dim
            weights = weights.expand(num_samples, -1, -1, -1, -1, -1)
            p_idxs = context.indices_out[..., None, None, None, None]  # make space for repetition dim
            p_idxs = p_idxs.expand(-1, -1, -1, num_repetitions, in_channels, in_channels)
            weights = weights.gather(dim=2, index=p_idxs)
            weights = weights.squeeze(2)

            # Index repetitions
            r_idxs = context.indices_repetition[..., None, None, None, None]
            r_idxs = r_idxs.expand(-1, out_features, -1, in_channels, in_channels)
            weights = weights.gather(dim=2, index=r_idxs)
            weights = weights.squeeze(2)

        else:
            # Index sums_out
            weights = weights.unsqueeze(0)  # make space for batch dim
            p_idxs = context.indices_out[..., None, None, None]  # make space for repetition dim
            weights = index_one_hot(weights, index=p_idxs, dim=2)
            assert weights.shape == (
                num_samples,
                out_features,
                num_repetitions,
                in_channels,
                in_channels,
            )

            # Index repetition
            r_idxs = context.indices_repetition.view(num_samples, 1, num_repetitions, 1, 1)
            weights = index_one_hot(weights, index=r_idxs, dim=2)

        # Check dimensions
        assert weights.shape == (num_samples, out_features, in_channels, in_channels)

        # Apply softmax to ensure they are proper probabilities
        weights = weights.view(num_samples, out_features, in_channels**2)
        log_weights = F.log_softmax(weights * context.temperature_sums, dim=2)

        # If evidence is given, adjust the weights with the likelihoods of the observed paths
        if self._is_input_cache_enabled and self._input_cache_left is not None:
            # Index repetition
            if context.is_differentiable:
                r_idxs = context.indices_repetition.view(num_samples, 1, 1, num_repetitions)
                lls_left = index_one_hot(self._input_cache_left, index=r_idxs, dim=-1).unsqueeze(3)
                lls_right = index_one_hot(self._input_cache_right, index=r_idxs, dim=-1).unsqueeze(2)
            else:
                r_idxs = context.indices_repetition[..., None, None, None]
                r_idxs = r_idxs.expand(-1, out_features, in_channels, -1)
                lls_left = self._input_cache_left.gather(index=r_idxs, dim=-1).squeeze(-1).unsqueeze(3)
                lls_right = self._input_cache_right.gather(index=r_idxs, dim=-1).squeeze(-1).unsqueeze(2)

            lls = (lls_left + lls_right).view(num_samples, out_features, in_channels**2)
            log_prior = log_weights
            log_posterior = log_prior + lls
            log_posterior = log_posterior - torch.logsumexp(log_posterior, dim=2, keepdim=True)
            log_weights = log_posterior

        # Sample/mpe from the logweights
        if not context.is_differentiable:  # Non-differentiable sampling
            if context.is_mpe:
                indices = log_weights.argmax(dim=2)
            else:
                # Create categorical distribution to sample from
                dist = torch.distributions.Categorical(logits=log_weights)
                indices = dist.sample()

            indices = self.unraveled_channel_indices[indices]
            indices = indices.view(num_samples, -1)

        else:  # Differentiable sampling
            indices = sample_categorical_differentiably(
                log_weights,
                dim=-1,
                is_mpe=context.is_mpe,
                hard=context.hard,
                tau=context.tau,
            )
            indices = indices.unsqueeze(-1)
            indices_a = (indices * self.unraveled_channel_indices_oh_0).sum(-2)
            indices_b = (indices * self.unraveled_channel_indices_oh_1).sum(-2)
            indices = torch.stack((indices_a, indices_b), dim=2)
            indices = indices.view(num_samples, -1, in_channels)

        context.indices_out = indices
        return context

    def _check_indices_repetition(self, context: SamplingContext):
        assert context.indices_repetition.shape[0] == context.indices_out.shape[0]
        if self.num_repetitions > 1 and context.indices_repetition is None:
            raise Exception(
                f"Sum layer has multiple repetitions (num_repetitions=={self.num_repetitions}) "
                f"but indices_repetition argument was None, expected a Long tensor size #samples."
            )
        if self.num_repetitions == 1 and context.indices_repetition is None:
            context.indices_repetition = torch.zeros(context.num_samples, dtype=torch.int, device=self.__device)

    def _enable_input_cache(self):
        """Enables the input cache. This will store the input in forward passes into
        `self.__input_cache`."""
        self._is_input_cache_enabled = True

    def _disable_input_cache(self):
        """Disables and clears the input cache."""
        self._is_input_cache_enabled = False
        self._input_cache_left = None
        self._input_cache_right = None

    def extra_repr(self):
        return "num_features={}, num_sums_in={}, num_sums_out={}, out_shape={}, " "weights_shape={}".format(
            self.num_features,
            self.num_sums_in,
            self.num_sums_out,
            self.out_shape,
            self.weights.shape,
        )


class EinsumMixingLayer(AbstractLayer):
    """
    A PyTorch module that implements a mixing layer using the Einstein summation convention.

    Attributes:
        weights (nn.Parameter): The learnable weights of the layer.
        num_sums_in (int): The number of input summation nodes.
        num_sums_out (int): The number of output summation nodes.
        out_features (int): The number of output features.
    """

    def __init__(
        self,
        num_features: int,
        num_sums_in: int,
        num_sums_out: int,
    ):
        """
        Creates a new EinsumMixingLayer.

        Args:
            num_features (int): The number of input features.
            num_sums_in (int): The number of input summation nodes.
            num_sums_out (int): The number of output summation nodes.
        """
        super().__init__(num_features, num_repetitions=1)

        self.num_sums_in = check_valid(num_sums_in, int, 1)
        self.num_sums_out = check_valid(num_sums_out, int, 1)
        self.out_features = num_features

        # Weights, such that each sumnode has its own weights
        ws = torch.randn(
            self.num_features,
            self.num_sums_out,
            self.num_sums_in,
        )
        self.weights = nn.Parameter(ws)

        # Necessary for sampling with evidence: Save input during forward pass.
        self._is_input_cache_enabled = False
        self._input_cache_left = None
        self._input_cache_right = None

    def forward(self, x):
        """Forward pass of the layer."""
        # Save input if input cache is enabled
        if self._is_input_cache_enabled:
            self._input_cache = x.clone()

        # Dimensions
        N, D, IC, R = x.size()

        probs_max = torch.max(x, dim=3, keepdim=True)[0]
        probs = torch.exp(x - probs_max)

        weights = self.weights
        weights = F.softmax(weights, dim=2)

        out = torch.einsum("bdoc,doc->bdo", probs, weights)
        lls = torch.log(out) + probs_max.squeeze(3)

        return lls

    def sample(
        self,
        num_samples: int = None,
        context: SamplingContext = None,
    ) -> SamplingContext:
        """
        Samples from the EinsumLayer.

        Args:
            num_samples (int, optional): The number of samples to generate. Defaults to None.
            context (SamplingContext, optional): The sampling context. Defaults to None.

        Returns:
            SamplingContext: The updated sampling context.
        """

    def sample(
        self,
        num_samples: int = None,
        context: SamplingContext = None,
    ) -> SamplingContext:
        # Sum weights are of shape: [W, H, IC, OC, R]
        # We now want to use `indices` to access one in_channel for each in_feature x num_sums_out block
        # index is of size in_feature
        weights = self.weights

        in_features, num_sums_out, num_sums_in = weights.shape
        num_samples = context.num_samples

        self._check_indices_repetition(context)

        if not context.is_differentiable:
            # Index with parent indices
            weights = weights.unsqueeze(0)  # make space for batch dim
            weights = weights.expand(num_samples, -1, -1, -1)
            p_idxs = context.indices_out.unsqueeze(-1).unsqueeze(-1)  # make space for repetition dim
            p_idxs = p_idxs.expand(-1, -1, -1, num_sums_in)
            weights = weights.gather(dim=2, index=p_idxs)
            # Drop dim which was selected via parent indices
            weights = weights.squeeze(2)
        else:
            # Index with parent indices
            weights = weights.unsqueeze(0)  # make space for batch dim
            p_idxs = context.indices_out.unsqueeze(-1)  # make space for repetition dim
            weights = index_one_hot(weights, index=p_idxs, dim=2)  # TODO: is 2 correct?

        # Check dimensions
        assert weights.shape == (num_samples, in_features, num_sums_in)

        log_weights = F.log_softmax(weights, dim=2)

        # If evidence is given, adjust the weights with the likelihoods of the observed paths
        if self._is_input_cache_enabled and self._input_cache is not None:
            # TODO: parallelize this with torch.gather
            for i in range(num_samples):
                # Reweight the i-th samples weights by its likelihood values at the correct
                # repetition
                log_weights[i, :, :] += self._input_cache[i, :, :, context.indices_repetition[i]]

        if not context.is_differentiable:
            if context.is_mpe:
                indices = log_weights.argmax(dim=2)
            else:
                # Create categorical distribution to sample from
                dist = torch.distributions.Categorical(logits=log_weights)

                indices = dist.sample()
        else:
            indices = sample_categorical_differentiably(
                log_weights,
                dim=2,
                is_mpe=context.is_mpe,
                hard=context.hard,
                tau=context.tau,
            )

        context.indices_out = indices
        return context

    def _check_indices_repetition(self, context: SamplingContext):
        assert context.indices_repetition.shape[0] == context.indices_out.shape[0]
        if self.num_repetitions > 1 and context.indices_repetition is None:
            raise Exception(
                f"Sum layer has multiple repetitions (num_repetitions=={self.num_repetitions}) but indices_repetition argument was None, expected a Long tensor size #samples."
            )
        if self.num_repetitions == 1 and context.indices_repetition is None:
            context.indices_repetition = torch.zeros(context.num_samples, dtype=torch.int, device=self.__device)

    def extra_repr(self):
        return "num_features={}, num_sums_in={}, num_sums_out={}".format(
            self.num_features, self.num_sums_in, self.num_sums_out, self.num_repetitions
        )


class MixingLayer(AbstractLayer):
    """
    A PyTorch module that implements a linear mixing layer.

    Attributes:
        weights (nn.Parameter): The learnable weights of the layer.
        num_sums_in (int): The number of input summation nodes.
        num_sums_out (int): The number of output summation nodes.
        out_features (int): The number of output features.
        _is_input_cache_enabled (bool): Whether the input cache is enabled.
        _input_cache_left (torch.Tensor): The left input cache.
        _input_cache_right (torch.Tensor): The right input cache.
        _bernoulli_dist (torch.distributions.Bernoulli): The Bernoulli distribution.
    """

    def __init__(
        self,
        num_features: int,
        num_sums_in: int,
        num_sums_out: int,
        dropout: float = 0.0,
    ):
        """
        Initializes an EinsumLayer instance.

        Args:
            num_features (int): Number of input and output features.
            num_sums_in (int): Number of input sum nodes.
            num_sums_out (int): Number of output sum nodes.
            dropout (float, optional): Dropout probability. Defaults to 0.0.
        """
        super().__init__(num_features, num_repetitions=1)

        self.num_sums_in = check_valid(num_sums_in, int, 1)
        self.num_sums_out = check_valid(num_sums_out, int, 1)
        self.out_features = num_features

        # Dropout
        self.dropout = check_valid(dropout, expected_type=float, lower_bound=0.0, upper_bound=1.0)
        self.dropout = nn.Parameter(torch.tensor(self.dropout), requires_grad=False)
        self._bernoulli_dist = torch.distributions.Bernoulli(probs=self.dropout)

        # Weights, such that each sumnode has its own weights
        ws = torch.randn(
            self.num_features,
            self.num_sums_out,
            self.num_sums_in,
        )
        self.weights = nn.Parameter(ws)

        # Necessary for sampling with evidence: Save input during forward pass.
        self._is_input_cache_enabled = False
        self._input_cache_left = None
        self._input_cache_right = None

    def _get_normalized_log_weights(self):
        return F.log_softmax(self.weights, dim=2)

    def forward(self, x):
        """Forward pass of the layer."""
        # Save input if input cache is enabled
        if self._is_input_cache_enabled:
            self._input_cache = x.clone()

        # Dimensions
        N, D, IC, R = x.size()

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
        log_weights = self._get_normalized_log_weights().unsqueeze(0)
        lls = torch.logsumexp(x + log_weights, dim=3)

        return lls

    def forward_tdi(self, log_exp_ch, log_var_ch, dropout_inference=None):
        # Save input if input cache is enabled
        if self._is_input_cache_enabled:
            self._input_cache = log_exp_ch.clone()

        # Dimensions
        N, D, IC, R = log_exp_ch.size()

        # Get log weights
        log_weights = self._get_normalized_log_weights().unsqueeze(0)

        # Prepare constants
        # If dropout at inference time is set, use this instead
        if dropout_inference is not None:
            log_q = np.log(1 - dropout_inference)
            log_p = np.log(dropout_inference)
        else:
            log_q = torch.log(1 - self.dropout)
            log_p = torch.log(self.dropout)

        # Expectation
        log_exp = log_q + torch.logsumexp(log_exp_ch + log_weights, dim=3)

        # Variance
        log_weights_sq = log_weights * 2
        log_exp_ch_sq = log_exp_ch * 2
        log_var_ch = log_var_ch

        log_var_plus_exp = torch.logsumexp(torch.stack((log_var_ch, log_exp_ch_sq + log_p), dim=-1), dim=-1)
        log_var = log_q + torch.logsumexp(log_weights_sq + log_var_plus_exp, dim=3)

        return log_exp, log_var

    def sample(
        self,
        num_samples: int = None,
        context: SamplingContext = None,
    ) -> SamplingContext:
        """
        Samples from the mixing layer.

        Args:
            num_samples (int): The number of samples to generate.
            context (SamplingContext): The sampling context.

        Returns:
            SamplingContext: The updated sampling context.
        """
        # raise NotImplementedError("Not yet implemented for MixingLayer")
        # Sum weights are of shape: [W, H, IC, OC, R]
        # We now want to use `indices` to access one in_channel for each in_feature x num_sums_out block
        # index is of size in_feature
        weights = self.weights

        in_features, num_sums_out, num_sums_in = weights.shape
        num_samples = context.num_samples

        self._check_indices_repetition(context)

        if not context.is_differentiable:
            # Index with parent indices
            weights = weights.unsqueeze(0)  # make space for batch dim
            weights = weights.expand(num_samples, -1, -1, -1)
            p_idxs = context.indices_out.unsqueeze(-1).unsqueeze(-1)  # make space for repetition dim
            p_idxs = p_idxs.expand(-1, -1, -1, num_sums_in)
            weights = weights.gather(dim=2, index=p_idxs)
            # Drop dim which was selected via parent indices
            weights = weights.squeeze(2)
        else:
            # Index with parent indices
            weights = weights.unsqueeze(0)  # make space for batch dim
            p_idxs = context.indices_out.unsqueeze(-1)  # make space for repetition dim
            weights = index_one_hot(weights, index=p_idxs, dim=2)  # TODO: is 2 correct?

        # Check dimensions
        assert weights.shape == (num_samples, in_features, num_sums_in)

        log_weights = F.log_softmax(weights, dim=2)

        # If evidence is given, adjust the weights with the likelihoods of the observed paths
        if self._is_input_cache_enabled and self._input_cache is not None:
            # TODO: parallelize this with torch.gather
            for i in range(num_samples):
                # Reweight the i-th samples weights by its likelihood values at the correct
                # repetition
                log_weights[i, :, :] += self._input_cache[i, :, :, context.indices_repetition[i]]

        if not context.is_differentiable:
            if context.is_mpe:
                indices = log_weights.argmax(dim=2)
            else:
                # Create categorical distribution to sample from
                dist = torch.distributions.Categorical(logits=log_weights)

                indices = dist.sample()
        else:
            indices = sample_categorical_differentiably(
                log_weights,
                is_mpe=context.is_mpe,
                dim=2,
                hard=context.hard,
                tau=context.tau,
            )

        context.indices_out = indices
        return context

    def _check_indices_repetition(self, context: SamplingContext):
        assert context.indices_repetition.shape[0] == context.indices_out.shape[0]
        if self.num_repetitions > 1 and context.indices_repetition is None:
            raise Exception(
                f"Sum layer has multiple repetitions (num_repetitions=={self.num_repetitions}) but indices_repetition argument was None, expected a Long tensor size #samples."
            )
        if self.num_repetitions == 1 and context.indices_repetition is None:
            context.indices_repetition = torch.zeros(context.num_samples, dtype=torch.int, device=self.__device)

    def extra_repr(self):
        return "num_features={}, num_sums_in={}, num_sums_out={}".format(
            self.num_features, self.num_sums_in, self.num_sums_out, self.num_repetitions
        )
