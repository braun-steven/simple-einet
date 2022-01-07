from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .layers import AbstractLayer
from .type_checks import check_valid
from .utils import SamplingContext, invert_permutation


class EinsumLayer(AbstractLayer):
    def __init__(
        self,
        in_shape: Tuple[int, int],
        num_sums_in: int,
        num_sums_out: int,
        num_repetitions: int,
        split_dim: str,
    ):
        super().__init__(in_shape, num_repetitions)

        self.num_sums_in = check_valid(num_sums_in, int, 1)
        self.num_sums_out = check_valid(num_sums_out, int, 1)
        self.cardinality = 2  # Fixed to binary graphs for now
        self._out_features = np.ceil(self.num_features / self.cardinality).astype(int)
        self._pad = 0
        assert split_dim in [
            "h",
            "w",
        ], f"Argument 'split_dim' must be either 'h' (height) or 'w' (width) but was {split_dim}"
        self.split_dim = split_dim

        # Compute correct output dimensions for width/height
        if split_dim == "h":
            out_height = self.in_shape[0] // self.cardinality
            out_width = self.in_shape[1]
        else:
            out_height = self.in_shape[0]
            out_width = self.in_shape[1] // self.cardinality

        self.out_shape = [out_height, out_width]

        # Construct left/right partition indices
        if split_dim == "h":
            indices = torch.arange(self.in_shape[0])
        else:
            indices = torch.arange(self.in_shape[1])

        self.left_idx = indices[0::2]
        self.right_idx = indices[1::2]

        self.inverse_idx = invert_permutation(torch.cat((self.left_idx, self.right_idx)))

        # Weights, such that each sumnode has its own weights
        weights = torch.randn(
            out_height,
            out_width,
            self.num_sums_in,
            self.num_sums_in,
            self.num_sums_out,
            self.num_repetitions,
        )
        self.weights = nn.Parameter(weights)

        # Create index map from flattened to coordinates (only needed in sampling)
        self.unraveled_channel_indices = nn.Parameter(
            torch.tensor(
                [(i, j) for i in range(self.num_sums_in) for j in range(self.num_sums_in)]
            ),
            requires_grad=False,
        )

        # Necessary for sampling with evidence: Save input during forward pass.
        self._is_input_cache_enabled = False
        self._input_cache_left = None
        self._input_cache_right = None

    def forward(self, x: torch.Tensor):
        """
        Einsum layer forward pass.

        Args:
            x: Input of shape [batch, num_features, channel].

        Returns:
            torch.Tensor: Output of shape [batch, ceil(num_features/2), channel * channel].
        """

        # Check if padding to next power of 2 is necessary
        if self.num_features != x.shape[1]:
            # Compute necessary padding to the next power of 2
            self._pad = 2 ** np.ceil(np.log2(x.shape[1])).astype(np.int) - x.shape[1]

            # Pad marginalized node
            x = F.pad(x, pad=[0, 0, 0, 0, 0, self._pad], mode="constant", value=0.0)

        # Dimensions
        N, H, W, C, R = x.size()
        assert self.num_sums_in == C
        assert self.num_features == H * W

        # left/right shape: [n, h/2, w, c, r] or [n, h, w/2, c, r]
        # if self.split_dim == "h":
        #     split_val = self.out_shape[0]
        #     left = x[:, split_val:]
        #     right = x[:, :split_val]
        # else:
        #     split_val = self.out_shape[1]
        #     left = x[:, :, split_val:]
        #     right = x[:, :, :split_val]
        if self.split_dim == "h":
            left = x[:, self.left_idx]
            right = x[:, self.right_idx]
        else:
            left = x[:, :, self.left_idx]
            right = x[:, :, self.right_idx]

        left_max = torch.max(left, dim=3, keepdim=True)[0]
        left_prob = torch.exp(left - left_max)
        right_max = torch.max(right, dim=3, keepdim=True)[0]
        right_prob = torch.exp(right - right_max)

        weights = self.weights

        weights_shape_orig = weights.shape
        weights = weights.view(
            *self.out_shape, self.num_sums_in ** 2, self.num_sums_out, self.num_repetitions
        )

        weights = F.softmax(weights, dim=2)
        weights = weights.view(weights_shape_orig)

        # Einsum operation for sum(product(x))
        # b: batch, i: left-in-sums, j: right-in-sums, w/h:width/height, o: out-sums, r: repetitions
        prob = torch.einsum("bhwir,bhwjr,hwijor->bhwor", left_prob, right_prob, weights)

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
        # We now want to use `indices` to access one in_channel for each in_feature x num_sums_out block
        # index is of size in_feature
        weights = self.weights

        height, width, num_sums_in, num_sums_in, num_sums_out, num_repetitions = weights.shape
        num_samples = context.num_samples

        self._check_indices_repetition(context)

        # Make space for batch dim
        weights = weights.unsqueeze(0)
        weights = weights.expand(
            num_samples, -1, -1, -1, -1, -1, -1
        )  # TODO: Make this somehow pretty?

        # Index with repetition indices
        r_idxs = context.indices_repetition.view(
            -1, 1, 1, 1, 1, 1, 1
        )  # TODO: Make this somehow pretty?
        r_idxs = r_idxs.expand(-1, height, width, num_sums_in, num_sums_in, num_sums_out, -1)
        weights = weights.gather(dim=-1, index=r_idxs)
        weights = weights.squeeze(-1)

        # Index with parent indices
        p_idxs = context.indices_out.view(-1, height, width, 1, 1, 1)
        p_idxs = p_idxs.expand(-1, -1, -1, num_sums_in, num_sums_in, -1)
        weights = weights.gather(dim=5, index=p_idxs)
        # Drop dim which was selected via parent indices
        weights = weights.squeeze(5)

        # Check dimensions
        assert weights.shape == (num_samples, height, width, num_sums_in, num_sums_in)

        # Apply softmax to ensure they are proper probabilities
        weights = weights.view(num_samples, height, width, num_sums_in ** 2)

        log_weights = F.log_softmax(weights, dim=-1)

        # If evidence is given, adjust the weights with the likelihoods of the observed paths
        if self._is_input_cache_enabled and self._input_cache_left is not None:
            # TODO: Adapt with torch.gather
            for i in range(num_samples):
                # Reweight the i-th samples weights by its likelihood values at the correct repetition
                lls_left = self._input_cache_left[
                    i, :, :, :, context.indices_repetition[i]
                ].unsqueeze(3)
                lls_right = self._input_cache_right[
                    i, :, :, :, context.indices_repetition[i]
                ].unsqueeze(2)
                lls = (lls_left + lls_right).view(height, width, num_sums_in ** 2)
                log_prior = log_weights[i]
                log_posterior = log_prior + lls
                log_posterior = log_posterior - torch.logsumexp(log_posterior, 2, keepdim=True)
                log_weights[i] = log_posterior

        if context.is_mpe:
            indices = log_weights.argmax(dim=-1)
        else:
            # Create categorical distribution to sample from
            dist = torch.distributions.Categorical(logits=log_weights)

            indices = dist.sample()

        # Invert the product operation: scatter sampled index from sum node across the binary partition
        # That is, since the forward pass either splits the input tensor in the height or width
        # dimension, we need to first translate the index in the range of [0, num_sums_in**2)
        # to the "square" of product node combinations by using the unraveled channel indices map, i.e.:
        # 0 -> [0,0], 1 -> [0, 1], 2 -> [1, 0], 3 -> [1, 1] in the case of a 2x2 input
        # From this output, the first element is the index for the left partition and the second
        # element is the index for the right partition
        # So something like
        #  [[1, 2]]
        # for a 2x2 input should be translated to
        #  [[0, 1]
        #   [1, 0]]
        #

        # Map indices from product output tensor to partition tensors (last dim is now of size 2)
        indices = self.unraveled_channel_indices[indices]
        if self.split_dim == "h":
            # Move partition dimension after the dimension at which we have split
            indices = indices.permute(0, 1, 3, 2)
            # Interleave tensors
            indices = indices.reshape(num_samples, 2 * height, width)
        else:
            indices = indices.permute(0, 1, 2, 3)
            # Interleave tensors
            indices = indices.reshape(num_samples, height, 2 * width)


        assert list(indices.shape[-2:]) == self.in_shape

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

    def _enable_input_cache(self):
        """Enables the input cache. This will store the input in forward passes into `self.__input_cache`."""
        self._is_input_cache_enabled = True

    def _disable_input_cache(self):
        """Disables and clears the input cache."""
        self._is_input_cache_enabled = False
        self._input_cache_left = None
        self._input_cache_right = None

    def __repr__(self):
        return "EinsumLayer(in_shape={}, out_shape={}, num_sums_in={}, num_sums_out={}, num_repetitions={}, split={})".format(
            self.in_shape, self.out_shape, self.num_sums_in, self.num_sums_out, self.num_repetitions, self.split_dim
        )


class EinsumMixingLayer(AbstractLayer):
    def __init__(
        self,
        in_shape: Tuple[int, int],
        num_sums_in: int,
        num_sums_out: int,
    ):
        super().__init__(in_shape, num_repetitions=1)

        self.num_sums_in = check_valid(num_sums_in, int, 1)
        self.num_sums_out = check_valid(num_sums_out, int, 1)
        self.out_shape = in_shape

        # Weights, such that each sumnode has its own weights
        ws = torch.randn(
            self.in_shape[0],
            self.in_shape[1],
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
        N, H, W, IC, R = x.size()
        oc = self.weights.size(2)

        probs_max = torch.max(x, dim=4, keepdim=True)[0]
        probs = torch.exp(x - probs_max)

        weights = self.weights
        weights = F.softmax(weights, dim=3)

        out = torch.einsum("bhwoc,hwoc->bhwo", probs, weights)
        lls = torch.log(out) + probs_max.squeeze(4)

        return lls

    def sample(
        self, num_samples: int = None, context: SamplingContext = None
    ) -> Union[SamplingContext, torch.Tensor]:
        # Sum weights are of shape: [W, H, IC, OC, R]
        # We now want to use `indices` to access one in_channel for each in_feature x num_sums_out block
        # index is of size in_feature
        weights = self.weights

        height, width, num_sums_out, num_sums_in = weights.shape
        num_samples = context.num_samples

        self._check_indices_repetition(context)

        # Index with parent indices
        weights = weights.unsqueeze(0)  # make space for batch dim
        weights = weights.expand(num_samples, -1, -1, -1, -1)
        p_idxs = context.indices_out.unsqueeze(-1).unsqueeze(-1)  # make space for repetition dim
        p_idxs = p_idxs.expand(-1, -1, -1, -1, num_sums_in)
        weights = weights.gather(dim=3, index=p_idxs)
        # Drop dim which was selected via parent indices
        weights = weights.squeeze(3)

        # Check dimensions
        assert weights.shape == (num_samples, height, width, num_sums_in)

        log_weights = F.log_softmax(weights, dim=3)

        # If evidence is given, adjust the weights with the likelihoods of the observed paths
        if self._is_input_cache_enabled and self._input_cache is not None:
            # TODO: parallelize this with torch.gather
            for i in range(num_samples):
                # Reweight the i-th samples weights by its likelihood values at the correct repetition
                log_weights[i, :, :, :] += self._input_cache[
                    i, :, :, :, context.indices_repetition[i]
                ]

        if context.is_mpe:
            indices = log_weights.argmax(dim=3)
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

    def __repr__(self):
        return "EinsumMixingLayer(in_shape={}, out_shape={}, num_sums_in={}, num_sums_out={}, num_repetitions={})".format(
            self.in_shape, self.out_shape, self.num_sums_in, self.num_sums_out, self.num_repetitions
        )
