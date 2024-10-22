import logging
from typing import List

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from simple_einet.abstract_layers import AbstractLayer
from simple_einet.sampling_utils import SamplingContext
from simple_einet.type_checks import check_valid

logger = logging.getLogger(__name__)


class RootProductLayer(AbstractLayer):
    def __init__(self, num_features: int, num_repetitions: int):
        super().__init__(num_features, num_repetitions)
        self.out_shape = f"(N, {self.num_features}, in_channels, {self.num_repetitions})"

    def forward(self, x: torch.Tensor):
        assert x.size(1) == self.num_features
        return x.sum(dim=1, keepdim=True)

    def sample(self, ctx: SamplingContext) -> SamplingContext:
        shape = [1] * ctx.indices_out.dim()
        shape[1] = self.num_features
        ctx.indices_out = ctx.indices_out.repeat(*shape)
        return ctx


class ProductLayer(AbstractLayer):
    """
    Product Node Layer that chooses k scopes as children for a product node.

    Attributes:
        cardinality (int): Number of random children for each product node.
        _conv_weights (torch.nn.Parameter): Convolution weights.
        _pad (int): Padding to the next power of 2.
        _out_features (int): Number of output features.
        out_shape (str): Output shape of the layer.
    """

    def __init__(self, in_features: int, cardinality: int, num_repetitions: int = 1):
        """
        Create a product node layer.

        Args:
            in_features (int): Number of input features.
            cardinality (int): Number of random children for each product node.
            num_repetitions (int, optional): Number of layer repetitions in parallel. Defaults to 1.
        """

        super().__init__(in_features, num_repetitions)

        self.cardinality = check_valid(cardinality, int, 1, in_features + 1)

        # Implement product as convolution
        self._conv_weights = nn.Parameter(torch.ones(1, 1, cardinality, 1, 1), requires_grad=False)
        self._pad = (self.cardinality - self.num_features % self.cardinality) % self.cardinality

        self._out_features = np.ceil(self.num_features / self.cardinality).astype(int)
        self.out_shape = f"(N, {self._out_features}, in_channels, {self.num_repetitions})"

    @property
    def _device(self):
        """Hack to obtain the current device, this layer lives on."""
        return self._conv_weights.device

    def forward(self, x: torch.Tensor):
        """
        Product layer forward pass.

        Args:
            x: Input of shape [batch, in_features, channel].

        Returns:
            torch.Tensor: Output of shape [batch, ceil(in_features/cardinality), channel].
        """
        # Only one product node
        if self.cardinality == x.shape[1]:
            return x.sum(1, keepdim=True)

        # Special case: if cardinality is 1 (one child per product node), this is a no-op
        if self.cardinality == 1:
            return x

        # Pad if in_features % cardinality != 0
        if self._pad > 0:
            x = F.pad(x, pad=(0, 0, 0, 0, 0, self._pad), value=0)

        # Dimensions
        n, d, c, r = x.size()
        d_out = d // self.cardinality

        # Use convolution with 3D weight tensor filled with ones to simulate the product node
        x = x.unsqueeze(1)  # Shape: [n, 1, d, c, r]
        result = F.conv3d(x, weight=self._conv_weights, stride=(self.cardinality, 1, 1))

        # Remove simulated channel
        result = result.squeeze(1)

        assert result.size() == (n, d_out, c, r)
        return result

    def sample(self, ctx: SamplingContext) -> SamplingContext:
        """Method to sample from this layer, based on the parents output.

        Args:
            ctx (SamplingContext): Sampling context.
        Returns:
            torch.Tensor: Index into tensor which paths should be followed.
                          Output should be of size: in_features, out_channels.
        """

        # If this is a root node
        if ctx.is_root:
            if self.num_repetitions == 1:
                # If there is only a single repetition, create new sampling context
                ctx.indices_out = torch.zeros(ctx.num_samples, 1, dtype=int, device=self._device)
                ctx.indices_repetition = torch.zeros(ctx.num_samples, dtype=int, device=self._device)
                return ctx
            else:
                raise Exception(
                    "Cannot start sampling from Product layer with num_repetitions > 1 and no context given."
                )
        else:
            # Repeat the parent indices, e.g. [0, 2, 3] -> [0, 0, 2, 2, 3, 3] depending on the cardinality
            indices = torch.repeat_interleave(ctx.indices_out, repeats=self.cardinality, dim=1)

            # Remove padding
            if self._pad:
                indices = indices[:, : -self._pad]

            ctx.indices_out = indices
            return ctx

    def extra_repr(self):
        return "num_features={}, cardinality={}, out_shape={}".format(
            self.num_features, self.cardinality, self.out_shape
        )


class CrossProductLayer(AbstractLayer):
    """
    Layerwise implementation of a RAT Product node.

    Builds the the combination of all children in two regions:
    res = []
    for n1 in R1, n2 in R2:
        res += [n1 * n2]

    TODO: Generalize to k regions (cardinality = k).

    Attributes:
        cardinality (int): Number of random children for each product node.
        _pad (int): Padding to the next power of 2.
        _out_features (int): Number of output features.
        out_shape (str): Output shape of the layer.
        _scopes (List[List[int]]): List of scopes for each product child.
    """

    def __init__(self, in_features: int, in_channels: int, num_repetitions: int = 1):
        """
        Create a rat product node layer.

        Args:
            in_features (int): Number of input features.
            in_channels (int): Number of input channels. This is only needed for the sampling pass.
        """

        super().__init__(in_features, num_repetitions)
        self.in_channels = check_valid(in_channels, int, 1)
        cardinality = 2  # Fixed to binary graphs for now
        self.cardinality = check_valid(cardinality, int, 2, in_features + 1)
        self._out_features = np.ceil(self.num_features / self.cardinality).astype(int)
        self._pad = 0

        # Collect scopes for each product child
        self._scopes = [[] for _ in range(self.cardinality)]

        # Create sequence of scopes
        scopes = np.arange(self.num_features)

        # For two consecutive scopes
        for i in range(0, self.num_features, self.cardinality):
            for j in range(cardinality):
                if i + j < in_features:
                    self._scopes[j].append(scopes[i + j])
                else:
                    # Case: d mod cardinality != 0 => Create marginalized nodes with prob 1.0
                    # Pad x in forward pass on the right: [n, d, c] -> [n, d+1, c] where index
                    # d+1 is the marginalized node (index "in_features")
                    self._scopes[j].append(self.num_features)

        # Transform into numpy array for easier indexing
        self._scopes = np.array(self._scopes)

        # Create index map from flattened to coordinates (only needed in sampling)
        self.unraveled_channel_indices = nn.Parameter(
            torch.tensor([(i, j) for i in range(self.in_channels) for j in range(self.in_channels)]),
            requires_grad=False,
        )

        self.out_shape = f"(N, {self._out_features}, {self.in_channels ** 2}, {self.num_repetitions})"

    @property
    def _device(self):
        """Hack to obtain the current device, this layer lives on."""
        return self.unraveled_channel_indices.device

    def forward(self, x: torch.Tensor):
        """
        Product layer forward pass.

        Args:
            x: Input of shape [batch, in_features, channel].

        Returns:
            torch.Tensor: Output of shape [batch, ceil(in_features/2), channel * channel].
        """
        # Check if padding to next power of 2 is necessary
        if self.num_features != x.shape[1]:
            # Compute necessary padding to the next power of 2
            self._pad = 2 ** np.ceil(np.log2(x.shape[1])).astype(np.int) - x.shape[1]

            # Pad marginalized node
            x = F.pad(x, pad=[0, 0, 0, 0, 0, self._pad], mode="constant", value=0.0)

        # Dimensions
        n, d, c, r = x.size()
        d_out = d // self.cardinality

        # Build outer sum, using broadcasting, this can be done with
        # modifying the tensor dimensions:
        # left: [n, d/2, c, r] -> [n, d/2, c, 1, r]
        # right: [n, d/2, c, r] -> [n, d/2, 1, c, r]
        left = x[:, self._scopes[0, :], :, :].unsqueeze(3)
        right = x[:, self._scopes[1, :], :, :].unsqueeze(2)

        # left + right with broadcasting: [n, d/2, c, 1, r] + [n, d/2, 1, c, r] -> [n, d/2, c, c, r]
        result = left + right

        # Put the two channel dimensions from the outer sum into one single dimension:
        # [n, d/2, c, c, r] -> [n, d/2, c * c, r]
        result = result.view(n, d_out, c * c, r)

        assert result.size() == (n, d_out, c * c, r)
        return result

    def sample(self, ctx: SamplingContext) -> SamplingContext:
        """Method to sample from this layer, based on the parents output.

        Args:
            ctx (SamplingContext): Sampling context.
        Returns:
            torch.Tensor: Index into tensor which paths should be followed.
                          Output should be of size: in_features, out_channels.
        """

        # If this is a root node
        if ctx.is_root:
            if self.num_repetitions == 1:
                # If there is only a single repetition, create new sampling context
                ctx.indices_out = torch.zeros(ctx.num_samples, 1, dtype=int, device=self._device)
                ctx.indices_repetition = torch.zeros(ctx.num_samples, dtype=int, device=self._device)
                return ctx
            else:
                raise Exception(
                    "Cannot start sampling from CrossProduct layer with num_repetitions > 1 and no context given."
                )

        # Map flattened indexes back to coordinates to obtain the chosen input_channel for each feature
        indices = self.unraveled_channel_indices[ctx.indices_out]
        indices = indices.view(indices.shape[0], -1)

        # Remove padding
        if self._pad:
            indices = indices[:, : -self._pad]

        ctx.indices_out = indices
        return ctx

    def extra_repr(self):
        return "num_features={}, out_shape={}".format(self.num_features, self.out_shape)
