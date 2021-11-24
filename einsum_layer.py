#!/usr/bin/env python3


from typing import Union
from utils import SamplingContext
import torch.nn.functional as F
import numpy as np
from torch import nn
import torch
from type_checks import check_valid
from layers import AbstractLayer


class EinsumLayer(AbstractLayer):
    def __init__(
        self,
        in_features: int,
        in_channels: int,
        out_channels: int,
        num_repetitions: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__(in_features, num_repetitions)

        self.in_channels = check_valid(in_channels, int, 1)
        self.out_channels = check_valid(out_channels, int, 1)
        cardinality = 2  # Fixed to binary graphs for now
        self.cardinality = check_valid(cardinality, int, 2, in_features + 1)
        self._out_features = np.ceil(self.in_features / self.cardinality).astype(int)
        self._pad = 0

        # Weights, such that each sumnode has its own weights
        ws = torch.randn(
            self.in_features // cardinality,
            self.in_channels,
            self.in_channels,
            self.out_channels,
            self.num_repetitions,
        )
        self.weights = nn.Parameter(ws)

        # Collect scopes for each product child
        self._scopes = [[] for _ in range(self.cardinality)]

        # Create sequence of scopes
        scopes = np.arange(self.in_features)

        # For two consecutive scopes
        for i in range(0, self.in_features, self.cardinality):
            for j in range(cardinality):
                if i + j < in_features:
                    self._scopes[j].append(scopes[i + j])
                else:
                    # Case: d mod cardinality != 0 => Create marginalized nodes with prob 1.0
                    # Pad x in forward pass on the right: [n, d, c] -> [n, d+1, c] where index
                    # d+1 is the marginalized node (index "in_features")
                    self._scopes[j].append(self.in_features)

        # Transform into numpy array for easier indexing
        self._scopes = np.array(self._scopes)

        # Create index map from flattened to coordinates (only needed in sampling)
        self.unraveled_channel_indices = nn.Parameter(
            torch.tensor(
                [(i, j) for i in range(self.in_channels) for j in range(self.in_channels)]
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

        self.out_shape = f"(N, {self._out_features}, {self.out_channels}, {self.num_repetitions})"

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

        # Check if padding to next power of 2 is necessary
        if self.in_features != x.shape[1]:
            # Compute necessary padding to the next power of 2
            self._pad = 2 ** np.ceil(np.log2(x.shape[1])).astype(np.int) - x.shape[1]

            # Pad marginalized node
            x = F.pad(x, pad=[0, 0, 0, 0, 0, self._pad], mode="constant", value=0.0)

        # Dimensions
        N, D, C, R = x.size()
        D_out = D // 2

        # Build outer sum, using broadcasting, this can be done with
        # modifying the tensor dimensions:
        # left: [n, d/2, c, r]
        # right: [n, d/2, c, r]
        left = x[:, self._scopes[0, :], :, :]
        right = x[:, self._scopes[1, :], :, :]

        left_max = torch.max(left, dim=2, keepdim=True)[0]
        left_prob = torch.exp(left - left_max)
        right_max = torch.max(right, dim=2, keepdim=True)[0]
        right_prob = torch.exp(right - right_max)

        weights = self.weights

        weights_shape_orig = weights.shape
        weights = weights.view(D_out, self.in_channels ** 2, *weights_shape_orig[3:])
        weights = F.softmax(weights, dim=1)
        weights = weights.view(weights_shape_orig)

        # Einsum operation for sum(product(x))
        # n: batch, i: left-channels, j: right-channels, d:features, o: output-channels, r: repetitions
        prob = torch.einsum("ndir,ndjr,dijor->ndor", left_prob, right_prob, weights)

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
        in_features, in_channels, in_channels, out_channels, num_repetitions = weights.shape
        num_samples = context.num_samples

        if context.is_root:
            assert (
                out_channels == 1 and num_repetitions == 1
            ), "Cannot start sampling from non-root layer."

            # Initialize rep indices
            context.repetition_indices = torch.zeros(num_samples, dtype=int, device=weights.device)

            # Select weights, repeat n times along the last dimension
            weights = weights[:, :, :, [0] * num_samples, 0]  # Shape: [D, IC2, IC2, N]

            # Move sample dimension to the first axis: [feat, channels, batch] -> [batch, feat, channels]
            weights = weights.permute(3, 0, 1, 2)  # Shape: [N, D, IC2, IC2]

        else:
            self._check_repetition_indices(context)

            tmp = torch.zeros(
                num_samples, in_features, in_channels, in_channels, device=weights.device
            )
            for i in range(num_samples):
                tmp[i, :, :, :] = weights[
                    range(in_features),
                    :,
                    :,
                    context.parent_indices[i],  # access the chosen output sum node
                    context.repetition_indices[i],  # access the chosen repetition
                ]

            weights = tmp

        # Check dimensions
        assert weights.shape == (num_samples, in_features, in_channels, in_channels)

        # Apply softmax to ensure they are proper probabilities
        weights = weights.view(num_samples, in_features, in_channels ** 2)

        log_weights = F.log_softmax(weights, dim=2)

        # If evidence is given, adjust the weights with the likelihoods of the observed paths
        if self._is_input_cache_enabled and self._input_cache_left is not None:
            for i in range(num_samples):
                # Reweight the i-th samples weights by its likelihood values at the correct repetition
                lls_left = self._input_cache_left[i, :, :, context.repetition_indices[i]].unsqueeze(
                    2
                )
                lls_right = self._input_cache_right[
                    i, :, :, context.repetition_indices[i]
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

        context.parent_indices = indices
        return context

    def _check_repetition_indices(self, context: SamplingContext):
        assert context.repetition_indices.shape[0] == context.parent_indices.shape[0]
        if self.num_repetitions > 1 and context.repetition_indices is None:
            raise Exception(
                f"Sum layer has multiple repetitions (num_repetitions=={self.num_repetitions}) but repetition_indices argument was None, expected a Long tensor size #samples."
            )
        if self.num_repetitions == 1 and context.repetition_indices is None:
            context.repetition_indices = torch.zeros(
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

    def __repr__(self):
        return "EinsumLayer(in_channels={}, in_features={}, out_channels={}, out_shape={}, weights_shape={})".format(
            self.in_channels,
            self.in_features,
            self.out_channels,
            self.out_shape,
            self.weights.shape,
        )
