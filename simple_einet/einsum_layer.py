from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from simple_einet.layers import AbstractLayer
from simple_einet.type_checks import check_valid
from simple_einet.utils import SamplingContext, index_one_hot, diff_sample_one_hot


class LinsumLayer(AbstractLayer):
    def __init__(
        self,
        num_features: int,
        num_sums_in: int,
        num_sums_out: int,
        num_repetitions: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__(num_features, num_repetitions)

        self.num_sums_in = check_valid(num_sums_in, int, 1)
        self.num_sums_out = check_valid(num_sums_out, int, 1)
        cardinality = 2  # Fixed to binary graphs for now
        self.cardinality = check_valid(cardinality, int, 2, num_features + 1)
        self.num_features_out = np.ceil(self.num_features / self.cardinality).astype(int)
        self._pad = 0

        ws = self._init_weights()

        self.weights = nn.Parameter(ws)

        # Dropout
        self.dropout = check_valid(dropout, expected_type=float, lower_bound=0.0, upper_bound=1.0)
        self._bernoulli_dist = torch.distributions.Bernoulli(probs=self.dropout)

        # Necessary for sampling with evidence: Save input during forward pass.
        self._is_input_cache_enabled = False
        self._input_cache_left = None
        self._input_cache_right = None

        self.out_shape = f"(N, {self.num_features_out}, {self.num_sums_out}, {self.num_repetitions})"

    def _init_weights(self):
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
            dropout_indices = self._bernoulli_dist.sample(prod_output.shape).bool()
            # TODO: this isn't allow, right? maybe add zeros with ninf at mask
            prod_output[dropout_indices] = np.NINF

        # Get log weights
        log_weights = self._get_normalized_log_weights().unsqueeze(0)
        # log_weights = F.log_softmax(self.weights, dim=1).unsqueeze(
        #     0
        # )  # 1 x D/2 x Sin x Sout x R
        prob = torch.logsumexp(prod_output + log_weights, dim=2)  # N x D/2 x Sout x R

        # Save input if input cache is enabled
        if self._is_input_cache_enabled:
            self._input_cache_left = left
            self._input_cache_right = right

        return prob

    def sample(
        self, num_samples: int, context: SamplingContext, differentiable=False
    ) -> Union[SamplingContext, torch.Tensor]:

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
            if context.is_mpe:
                raise NotImplementedError()
            else:
                indices = diff_sample_one_hot(
                    log_weights,
                    dim=-1,
                    mode="sample",
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
        super().__init__(num_features, num_repetitions)

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

    def sample(
        self, num_samples: int, context: SamplingContext, differentiable=False
    ) -> Union[SamplingContext, torch.Tensor]:

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

        # Clone for discrete validity check
        # log_weights_disc = log_weights.clone()

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
            if context.is_mpe:
                raise NotImplementedError()
            else:
                indices = diff_sample_one_hot(
                    log_weights,
                    dim=-1,
                    mode="sample",
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
    def __init__(
        self,
        num_features: int,
        num_sums_in: int,
        num_sums_out: int,
    ):
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
        differentiable=False,
    ) -> Union[SamplingContext, torch.Tensor]:
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
            if context.is_mpe:
                raise NotImplementedError
            else:
                indices = diff_sample_one_hot(
                    log_weights,
                    mode="sample",
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
