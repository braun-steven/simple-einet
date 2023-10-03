from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from simple_einet.abstract_layers import AbstractSumLayer, logits_to_log_weights
from simple_einet.sampling_utils import SamplingContext, sample_categorical_differentiably, index_one_hot


def logsumexp(tensors, mask=None, dim=-1):
    """
    Source: https://github.com/pytorch/pytorch/issues/32097

    Logsumexp with custom scalar mask to allow for negative values in the sum.

    Args:
        tensors (Tensor, List[Tensor]): The tensors to sum.
        mask (Tensor, optional): The mask to apply to the sum. Defaults to None.
        dim (int, optional): The dimension to sum over. Defaults to -1.

    Returns:
        Tensor: The summed tensor.
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


class EinsumLayer(AbstractSumLayer):
    # Fixed to binary graphs
    cardinality = 2

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
        super().__init__(
            num_features=num_features,
            num_repetitions=num_repetitions,
            num_sums_in=num_sums_in,
            num_sums_out=num_sums_out,
        )

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

        self.out_shape = f"(N, {self.num_features_out}, {self.num_sums_out}, {self.num_repetitions})"

    @property
    def num_features_out(self) -> int:
        return self.num_features // EinsumLayer.cardinality

    def weight_shape(self) -> Tuple[int, ...]:
        return (
            self.num_features_out,
            self.num_sums_out,
            self.num_repetitions,
            self.num_sums_in,
            self.num_sums_in,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Einsum layer forward pass.

        Args:
            x: Input of shape [batch, in_features, channel].

        Returns:
            Tensor: Output of shape [batch, ceil(in_features/2), channel * channel].
        """

        # Apply dropout: Set random sum node children to 0 (-inf in log domain)
        if self.dropout > 0.0 and self.training:
            dropout_indices = self._bernoulli_dist.sample(x.shape).bool()
            x[dropout_indices] = np.NINF

        # Dimensions
        N, D, C, R = x.size()
        D_out = D // 2

        # Get left and right partition probs
        left_log_prob = x[:, 0::2]
        right_log_prob = x[:, 1::2]

        # Prepare for LogEinsumExp trick (see paper for details)
        left_prob_max = torch.max(left_log_prob, dim=2, keepdim=True)[0]
        left_prob = torch.exp(left_log_prob - left_prob_max)
        right_prob_max = torch.max(right_log_prob, dim=2, keepdim=True)[0]
        right_prob = torch.exp(right_log_prob - right_prob_max)

        # Project weights into valid space
        logits = self.logits.view(D_out, self.num_sums_out, self.num_repetitions, -1)
        weights = F.softmax(logits, dim=-1)
        weights = weights.view(self.weight_shape())

        # Einsum operation for sum(product(x))
        # n: batch, i: left-channels, j: right-channels, d:features, o: output-channels, r: repetitions
        prob = torch.einsum("ndir,ndjr,dorij->ndor", left_prob, right_prob, weights)

        # LogEinsumExp trick, re-add the max
        prob = torch.log(prob) + left_prob_max + right_prob_max

        # Save input if input cache is enabled
        if self._is_input_cache_enabled:
            self._input_cache["left"] = left_log_prob
            self._input_cache["right"] = right_log_prob

        return prob

    def _sample_from_weights(self, ctx: SamplingContext, log_weights: Tensor):
        if ctx.is_differentiable:  # Differentiable sampling
            indices = sample_categorical_differentiably(
                dim=-1, is_mpe=ctx.is_mpe, hard=ctx.hard, tau=ctx.tau, log_weights=log_weights
            )
            indices = indices.unsqueeze(-1)
            indices_a = (indices * self.unraveled_channel_indices_oh_0).sum(-2)
            indices_b = (indices * self.unraveled_channel_indices_oh_1).sum(-2)
            indices = torch.stack((indices_a, indices_b), dim=2)
            indices = indices.view(ctx.num_samples, -1, self.num_sums_in)

        else:  # Non-differentiable sampling
            if ctx.is_mpe:
                indices = log_weights.argmax(dim=2)
            else:
                # Create categorical distribution to sample from
                dist = torch.distributions.Categorical(logits=log_weights)
                indices = dist.sample()
                # indices = SIMPLE(log_weights=log_weights, dim=-1)

            indices = self.unraveled_channel_indices[indices]
            indices = indices.view(ctx.num_samples, -1)
        return indices

    def _condition_weights_on_evidence(self, ctx: SamplingContext, log_weights: Tensor) -> Tensor:
        # Extract input cache
        input_cache_left = self._input_cache["left"]
        input_cache_right = self._input_cache["right"]

        # Index repetition
        if ctx.is_differentiable:
            r_idxs = ctx.indices_repetition.view(ctx.num_samples, 1, 1, self.num_repetitions)
            lls_left = index_one_hot(input_cache_left, index=r_idxs, dim=-1).unsqueeze(3)
            lls_right = index_one_hot(input_cache_right, index=r_idxs, dim=-1).unsqueeze(2)
        else:
            r_idxs = ctx.indices_repetition[..., None, None, None]
            r_idxs = r_idxs.expand(-1, self.num_features_out, self.num_sums_in, -1)
            lls_left = input_cache_left.gather(index=r_idxs, dim=-1).squeeze(-1).unsqueeze(3)
            lls_right = input_cache_right.gather(index=r_idxs, dim=-1).squeeze(-1).unsqueeze(2)
        lls = (lls_left + lls_right).view(ctx.num_samples, self.num_features_out, self.num_sums_in**2)
        log_prior = log_weights
        log_posterior = log_prior + lls
        log_posterior = log_posterior - torch.logsumexp(log_posterior, dim=2, keepdim=True)
        log_weights = log_posterior
        return log_weights

    def _select_weights(self, ctx: SamplingContext, logits: Tensor) -> Tensor:
        if ctx.is_differentiable:
            # Index sums_out
            logits = logits.unsqueeze(0)  # make space for batch dim
            p_idxs = ctx.indices_out[..., None, None, None]  # make space for repetition dim
            logits = index_one_hot(logits, index=p_idxs, dim=2)
            assert logits.shape == (
                ctx.num_samples,
                self.num_features_out,
                self.num_repetitions,
                self.num_sums_in,
                self.num_sums_in,
            )

            # Index repetition
            r_idxs = ctx.indices_repetition.view(ctx.num_samples, 1, self.num_repetitions, 1, 1)
            logits = index_one_hot(logits, index=r_idxs, dim=2)

        else:
            # Index sums_out
            logits = logits.unsqueeze(0)  # make space for batch dim
            logits = logits.expand(ctx.num_samples, -1, -1, -1, -1, -1)
            p_idxs = ctx.indices_out[..., None, None, None, None]
            p_idxs = p_idxs.expand(-1, -1, -1, self.num_repetitions, self.num_sums_in, self.num_sums_in)
            logits = logits.gather(dim=2, index=p_idxs)
            logits = logits.squeeze(2)

            # Index repetitions
            r_idxs = ctx.indices_repetition[..., None, None, None, None]
            r_idxs = r_idxs.expand(-1, self.num_features_out, -1, self.num_sums_in, self.num_sums_in)
            logits = logits.gather(dim=2, index=r_idxs)
            logits = logits.squeeze(2)

        # Check dimensions
        assert logits.shape == (ctx.num_samples, self.num_features_out, self.num_sums_in, self.num_sums_in)

        # Project weights into valid space
        log_weights = logits_to_log_weights(logits.view(*logits.shape[:-2], self.num_sums_in**2), dim=-1, temperature=ctx.temperature_sums)

        return log_weights

    def extra_repr(self):
        return "num_features={}, num_sums_in={}, num_sums_out={}, out_shape={}, " "weight_shape={}".format(
            self.num_features,
            self.num_sums_in,
            self.num_sums_out,
            self.out_shape,
            self.weight_shape(),
        )


