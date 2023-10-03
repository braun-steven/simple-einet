from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import nn, Tensor

from simple_einet.sampling_utils import SamplingContext
from simple_einet.type_checks import check_valid
from torch.nn import functional as F


class AbstractLayer(nn.Module, ABC):
    """
    This is the abstract base class for all layers in the SPN.
    """

    def __init__(self, num_features: int, num_repetitions: int = 1):
        """
        Create an abstract layer.

        Args:
            num_features (int): Number of input features.
            num_repetitions (int, optional): Number of layer repetitions in parallel. Defaults to 1.
        """
        super().__init__()
        self.num_features = check_valid(num_features, int, 1)
        self.num_repetitions = check_valid(num_repetitions, int, 1)

    @abstractmethod
    def sample(self, ctx: SamplingContext) -> SamplingContext:
        """
        Sample from this layer.

        Args:
            ctx (SamplingContext, optional): Sampling context. Defaults to None.

        Returns:
            SamplingContext: Sampling context with updated information.
        """
        pass


def logits_to_log_weights(logits: torch.Tensor, dim: int, temperature: float = 1.0) -> torch.Tensor:
    """Project logits (unnormalized log probabilities) to log weights (normalized log probabilities).

    Args:
        logits (torch.Tensor): The unnormalized log probabilities.
        dim (int): The dimension along which the projection is performed.
        temperature (float): The temperature of the projection.

    Returns:
        torch.Tensor: The normalized log probabilities.
    """
    return F.log_softmax(logits / temperature, dim=dim)


class AbstractSumLayer(AbstractLayer):
    """
    This is the abstract base class for all kinds of sum layers in the circuit.

    Sum layers need to implement the following methods:
    - weight_shape: Returns the shape of the layer's weights as a tuple of integers.
    - logits_to_log_weights: Project logits (unnormalized log probabilities) to log weights (normalized log probabilities).
    - _select_weights: Selects the appropriate weights tensor based on the given context during sampling.
    - _sample_from_weights: Samples indices from a categorical distribution defined by the given log weights during sampling.
    - _condition_weights_on_evidence_: Computes the posterior distribution over weights given the input evidence, conditioned on the evidence likelihoods.


    Args:
        num_features (int): Number of input features.
        num_sums_in (int): Number of input sums.
        num_sums_out (int): Number of output sums.
        num_repetitions (int, optional): Number of layer repetitions in parallel. Defaults to 1.
        dropout (float, optional): Dropout probability. Defaults to 0.0.
    """

    def __init__(
        self, num_features: int, num_sums_in: int, num_sums_out: int, num_repetitions: int, dropout: float = 0.0
    ):
        super().__init__(num_features=num_features, num_repetitions=num_repetitions)
        self.num_sums_in = check_valid(num_sums_in, int, 1)
        self.num_sums_out = check_valid(num_sums_out, int, 1)

        # Weights of this layer as unnormalized log probabilities
        self.logits = nn.Parameter(torch.log(torch.rand(self.weight_shape())))

        # Dropout
        self.dropout = nn.Parameter(torch.tensor(check_valid(dropout, float, 0.0, 1.0)), requires_grad=False)
        self._bernoulli_dist = torch.distributions.Bernoulli(probs=self.dropout)

        # Input cache
        self._is_input_cache_enabled = False
        self._input_cache: dict[str, Tensor] = {}

    @property
    @abstractmethod
    def num_features_out(self) -> int:
        """Returns the number of output features of this layer."""
        pass

    def _enable_input_cache(self):
        """Enables the input cache. This will store the input in forward passes into
        `self.__input_cache`."""
        self._is_input_cache_enabled = True

    def _disable_input_cache(self):
        """Disables and clears the input cache."""
        self._is_input_cache_enabled = False
        self._input_cache.clear()

    @property
    def _device(self):
        """Returns the device of the layer's logits."""
        return self.logits.device

    def _check_repetition_indices(self, ctx: SamplingContext):
        if self.num_repetitions == 1 and ctx.indices_repetition is None:
            ctx.indices_repetition = torch.zeros(ctx.num_samples, dtype=torch.long, device=self._device)

        assert ctx.indices_repetition.shape[0] == ctx.indices_out.shape[0]
        if self.num_repetitions > 1 and ctx.indices_repetition is None:
            raise Exception(
                f"Layer has multiple repetitions (num_repetitions=={self.num_repetitions}) "
                f"but indices_repetition argument was None, expected a Long tensor size #samples."
            )

    def _check_context_shapes(self, ctx: SamplingContext):
        assert ctx.indices_out.shape[0] == ctx.num_samples
        assert ctx.indices_out.shape[1] == self.num_features_out
        assert ctx.indices_repetition.shape[0] == ctx.num_samples

        if ctx.is_differentiable:
            assert ctx.indices_out.dim() == 3
            assert ctx.indices_out.shape[2] == self.num_sums_out

            assert ctx.indices_repetition.dim() == 2
            assert ctx.indices_repetition.shape[1] == self.num_repetitions
        else:
            assert ctx.indices_out.dim() == 2
            assert ctx.indices_repetition.dim() == 1

    def sample(self, ctx: SamplingContext) -> SamplingContext:
        self._check_context_shapes(ctx)
        self._check_repetition_indices(ctx)

        # Select weights of this layer based on parent sampling path
        log_weights = self._select_weights(ctx, self.logits)

        # If evidence is given, adjust the weights with the likelihoods of the observed paths
        if self._is_input_cache_enabled and len(self._input_cache) > 0:
            log_weights = self._condition_weights_on_evidence(ctx, log_weights)

        # Sample/mpe from the logweights
        indices = self._sample_from_weights(ctx, log_weights)

        ctx.indices_out = indices
        return ctx

    @abstractmethod
    def weight_shape(self) -> Tuple[int, ...]:
        """Returns the shape of the layer's weights as a tuple of integers."""
        pass


    @abstractmethod
    def _select_weights(self, context: SamplingContext, logits: torch.Tensor) -> torch.Tensor:
        """
        Selects the appropriate logits tensor based on the given context and project logits to log weights.

        Args:
            context (Context): The context object containing information about the current computation.
            logits (torch.Tensor): The weights tensor to select from.

        Returns:
            torch.Tensor: The selected and projected log weights tensor.
        """
        pass

    @abstractmethod
    def _sample_from_weights(self, context: SamplingContext, log_weights: torch.Tensor) -> torch.Tensor:
        """
        Samples indices from a categorical distribution defined by the given log weights.

        Args:
            context (Context): The context object that specifies the sampling mode and parameters.
            log_weights (torch.Tensor): The log weights of the categorical distribution.

        Returns:
            torch.Tensor: The sampled indices, reshaped as a tensor of shape (context.num_samples, -1, self.num_sums_in).
        """
        pass

    @abstractmethod
    def _condition_weights_on_evidence(self, context: SamplingContext, log_weights: torch.Tensor):
        """
        Computes the posterior distribution over weights given the input evidence, conditioned on the evidence likelihoods.

        Args:
            context (Context): The context object containing information about the current computation.
            log_weights (torch.Tensor): The log weights of the current computation.
        """
        pass
