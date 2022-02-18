from typing import Tuple, Union
import torch.nn.functional as F
import numpy as np
from torch import nn
import torch

from simple_einet.utils import SamplingContext
from simple_einet.type_checks import check_valid
from simple_einet.layers import AbstractLayer


class EinsumLayer(AbstractLayer):
    def __init__(
        self,
        num_features: int,
        num_sums_in: int,
        num_sums_out: int,
        num_repetitions: int = 1,
        dropout: float = 0.0,
        use_em: bool = False,
    ):
        super().__init__(num_features, num_repetitions, use_em)

        self.num_sums_in = check_valid(num_sums_in, int, 1)
        self.num_sums_out = check_valid(num_sums_out, int, 1)
        cardinality = 2  # Fixed to binary graphs for now
        self.cardinality = check_valid(cardinality, int, 2, num_features + 1)
        self.num_features_out = np.ceil(self.num_features / self.cardinality).astype(int)
        self._pad = 0

        # Weights, such that each sumnode has its own weights
        ws = torch.randn(
            self.num_features // cardinality,
            self.num_sums_out,
            self.num_repetitions,
            self.num_sums_in,
            self.num_sums_in,
        )
        # Project weights into valid space
        ws_shape_orig = ws.shape
        ws = ws.view(ws.shape[0], self.num_sums_out, self.num_repetitions, -1)
        ws = F.softmax(ws, dim=-1)
        ws = ws.view(ws_shape_orig)

        self.normalization_dims = [3, 4]

        self.weights = nn.Parameter(ws)

        # Create index map from flattened to coordinates (only needed in sampling)
        self.unraveled_channel_indices = nn.Parameter(
            torch.tensor(
                [(i, j) for i in range(self.num_sums_in) for j in range(self.num_sums_in)]
            ),
            requires_grad=False,
        )

        # Dropout
        self.dropout = check_valid(dropout, expected_type=float, lower_bound=0.0, upper_bound=1.0)
        self._bernoulli_dist = torch.distributions.Bernoulli(probs=self.dropout)

        # Necessary for sampling with evidence: Save input during forward pass.
        self._is_input_cache_enabled = False
        self._input_cache_left = None
        self._input_cache_right = None

        self.out_shape = (
            f"(N, {self.num_features_out}, {self.num_sums_out}, {self.num_repetitions})"
        )

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

        weights = self.weights

        # Project weights into valid space
        if not self.use_em:
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

    def sample(
        self, num_samples: int, context: SamplingContext
    ) -> Union[SamplingContext, torch.Tensor]:

        # Sum weights are of shape: [D, IC//2, IC//2, OC, R]
        # We now want to use `indices` to access one in_channel for each in_feature x out_channels block
        # index is of size in_feature
        weights = self.weights
        in_features, out_channels, num_repetitions, in_channels, in_channels = weights.shape
        num_samples = context.num_samples

        self._check_indices_repetition(context)

        tmp = torch.zeros(num_samples, in_features, in_channels, in_channels, device=weights.device)
        for i in range(num_samples):
            tmp[i, :, :, :] = weights[
                range(in_features),
                context.indices_out[i],  # access the chosen output sum node
                context.indices_repetition[i],  # access the chosen repetition
                :,
                :,
            ]

        weights = tmp

        # Check dimensions
        assert weights.shape == (num_samples, in_features, in_channels, in_channels)

        # Apply softmax to ensure they are proper probabilities
        weights = weights.view(num_samples, in_features, in_channels ** 2)

        if self.use_em:
            log_weights = torch.log(weights * context.temperature_sums)
        else:
            # Project with softmax first
            log_weights = F.log_softmax(weights * context.temperature_sums, dim=2)

        # If evidence is given, adjust the weights with the likelihoods of the observed paths
        if self._is_input_cache_enabled and self._input_cache_left is not None:
            for i in range(num_samples):
                # Reweight the i-th samples weights by its likelihood values at the correct repetition
                lls_left = self._input_cache_left[i, :, :, context.indices_repetition[i]].unsqueeze(
                    2
                )
                lls_right = self._input_cache_right[
                    i, :, :, context.indices_repetition[i]
                ].unsqueeze(1)
                lls = (lls_left + lls_right).view(in_features, in_channels ** 2)
                log_prior = log_weights[i, :, :]
                log_posterior = log_prior + lls
                log_posterior = log_posterior - torch.logsumexp(log_posterior, 1, keepdim=True)
                log_weights[i] = log_posterior

        if context.is_mpe:
            indices = log_weights.argmax(dim=2)
        else:
            # Create categorical distribution to sample from
            dist = torch.distributions.Categorical(logits=log_weights)

            indices = dist.sample()

        indices = self.unraveled_channel_indices[indices]
        indices = indices.view(num_samples, -1)

        context.indices_out = indices
        return context

    def em_purge(self):
        """Discard em statistics."""
        em_purge(self.weights)

    def em_update(self, stepsize: float):
        if not self.use_em:
            raise AssertionError("em_update called while _use_em==False.")
        em_update(self.weights, stepsize=stepsize, normalization_dims=self.normalization_dims)

    def _check_indices_repetition(self, context: SamplingContext):
        assert context.indices_repetition.shape[0] == context.indices_out.shape[0]
        if self.num_repetitions > 1 and context.indices_repetition is None:
            raise Exception(
                f"Sum layer has multiple repetitions (num_repetitions=={self.num_repetitions}) but indices_repetition argument was None, expected a Long tensor size #samples."
            )
        if self.num_repetitions == 1 and context.indices_repetition is None:
            context.indices_repetition = torch.zeros(
                context.num_samples, dtype=int, device=self.__device
            )

    def _enable_input_cache(self):
        """Enables the input cache. This will store the input in forward passes into `self.__input_cache`."""
        self._is_input_cache_enabled = True

    def _disable_input_cache(self):
        """Disables and clears the input cache."""
        self._is_input_cache_enabled = False
        self._input_cache_left = None
        self._input_cache_right = None

    def extra_repr(self):
        return "num_features={}, num_sums_in={}, num_sums_out={}, out_shape={}".format(
            self.num_features,
            self.num_sums_in,
            self.num_sums_out,
            self.out_shape,
        )


class EinsumMixingLayer(AbstractLayer):
    def __init__(
        self,
        num_features: int,
        num_sums_in: int,
        num_sums_out: int,
        use_em=False,
    ):
        super().__init__(num_features, num_repetitions=1, use_em=use_em)

        self.num_sums_in = check_valid(num_sums_in, int, 1)
        self.num_sums_out = check_valid(num_sums_out, int, 1)
        self.out_features = num_features

        # Weights, such that each sumnode has its own weights
        ws = torch.randn(
            self.num_features,
            self.num_sums_out,
            self.num_sums_in,
        )
        if use_em:
            ws_shape_orig = ws.shape
            ws = ws.view(ws.shape[0], self.num_sums_out, -1)
            ws = F.softmax(ws, dim=-1)
            ws = ws.view(ws_shape_orig)

        self.normalization_dims = [2]
        self.weights = nn.Parameter(ws)

        # Necessary for sampling with evidence: Save input during forward pass.
        self._is_input_cache_enabled = False
        self._input_cache_left = None
        self._input_cache_right = None

    def forward(self, x):
        # Save input if input cache is enabled
        if self._is_input_cache_enabled:
            self._input_cache = x.clone()

        # Dimensions
        N, D, IC, R = x.size()

        probs_max = torch.max(x, dim=3, keepdim=True)[0]
        probs = torch.exp(x - probs_max)

        weights = self.weights

        # Project weights
        if not self.use_em:
            weights = F.softmax(weights, dim=2)

        out = torch.einsum("bdoc,doc->bdo", probs, weights)
        lls = torch.log(out) + probs_max.squeeze(3)

        return lls

    def sample(
        self, num_samples: int = None, context: SamplingContext = None
    ) -> Union[SamplingContext, torch.Tensor]:
        # Sum weights are of shape: [W, H, IC, OC, R]
        # We now want to use `indices` to access one in_channel for each in_feature x num_sums_out block
        # index is of size in_feature
        weights = self.weights

        in_features, num_sums_out, num_sums_in = weights.shape
        num_samples = context.num_samples

        self._check_indices_repetition(context)

        # Index with parent indices
        weights = weights.unsqueeze(0)  # make space for batch dim
        weights = weights.expand(num_samples, -1, -1, -1)
        p_idxs = context.indices_out.unsqueeze(-1).unsqueeze(-1)  # make space for repetition dim
        p_idxs = p_idxs.expand(-1, -1, -1, num_sums_in)
        weights = weights.gather(dim=2, index=p_idxs)
        # Drop dim which was selected via parent indices
        weights = weights.squeeze(2)

        # Check dimensions
        assert weights.shape == (num_samples, in_features, num_sums_in)

        if self.use_em:
            log_weights = torch.log(weights)
        else:
            log_weights = F.log_softmax(weights, dim=2)

        # If evidence is given, adjust the weights with the likelihoods of the observed paths
        if self._is_input_cache_enabled and self._input_cache is not None:
            # TODO: parallelize this with torch.gather
            for i in range(num_samples):
                # Reweight the i-th samples weights by its likelihood values at the correct repetition
                log_weights[i, :, :] += self._input_cache[i, :, :, context.indices_repetition[i]]

        if context.is_mpe:
            indices = log_weights.argmax(dim=2)
        else:
            # Create categorical distribution to sample from
            dist = torch.distributions.Categorical(logits=log_weights)

            indices = dist.sample()

        context.indices_out = indices
        return context

    def _check_indices_repetition(self, context: SamplingContext):
        assert context.indices_repetition.shape[0] == context.indices_out.shape[0]
        if self.num_repetitions > 1 and context.indices_repetition is None:
            raise Exception(
                f"Sum layer has multiple repetitions (num_repetitions=={self.num_repetitions}) but indices_repetition argument was None, expected a Long tensor size #samples."
            )
        if self.num_repetitions == 1 and context.indices_repetition is None:
            context.indices_repetition = torch.zeros(
                context.num_samples, dtype=int, device=self.__device
            )

    def extra_repr(self):
        return "num_features={}, num_sums_in={}, num_sums_out={}".format(
            self.num_features, self.num_sums_in, self.num_sums_out, self.num_repetitions
        )

    def em_purge(self):
        """Discard em statistics."""
        em_purge(self.weights)

    def em_update(self, stepsize: float):
        if not self.use_em:
            raise AssertionError("em_update called while _use_em==False.")
        em_update(self.weights, stepsize=stepsize, normalization_dims=self.normalization_dims)


def em_purge(weights: torch.Tensor):
    """Discard em statistics."""
    weights.grad = None


def em_update(weights, stepsize, normalization_dims):
    """
    Source: Mostly taken/adapted from the official EinsumNetworks implementation.

    Do an EM update. If the setting is online EM (online_em_stepsize is not None), then this function does nothing,
    since updates are triggered automatically. Thus, leave the private parameter _triggered alone.

    :param _triggered: for internal use, don't set
    :return: None
    """

    with torch.no_grad():
        n = weights.grad * weights.data

        p = torch.clamp(n, 1e-16)
        p = p / (p.sum(normalization_dims, keepdim=True))
        weights.data = (1.0 - stepsize) * weights + stepsize * p

        weights.data = torch.clamp(weights, 1e-16)
        weights.data = weights / (weights.sum(normalization_dims, keepdim=True))
        weights.grad = None
