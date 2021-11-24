#!/usr/bin/env python3

import logging
from dataclasses import dataclass
from typing import Dict, List, Type

import numpy as np
import torch
from torch import nn

from clipper import DistributionClipper
from factorized_leaf_layer import FactorizedLeaf
from distributions import (
    Leaf,
    RatNormal,
    truncated_normal_,
)
from layers import Sum
from einsum_layer import EinsumLayer
from type_checks import check_valid
from utils import SamplingContext, provide_evidence

logger = logging.getLogger(__name__)


@dataclass
class EinetConfig:
    """
    Class for keeping the RatSpn config. Parameter names are according to the original RatSpn paper.

    in_features: int  # Number of input features
    D: int  # Tree depth
    S: int  # Number of sum nodes at each layer
    I: int  # Number of distributions for each scope at the leaf layer
    R: int  # Number of repetitions
    C: int  # Number of root heads / Number of classes
    dropout: float  # Dropout probabilities for leafs and sum layers
    leaf_base_class: Type  # Type of the leaf base class (Normal, Bernoulli, etc)
    leaf_base_kwargs: Dict  # Parameters for the leaf base class
    """

    in_features: int = None
    D: int = None
    S: int = None
    I: int = None
    R: int = None
    C: int = None
    dropout: float = None
    leaf_base_class: Type = None
    leaf_base_kwargs: Dict = None

    @property
    def F(self):
        """Alias for in_features."""
        return self.in_features

    @F.setter
    def F(self, in_features):
        """Alias for in_features."""
        self.in_features = in_features

    def assert_valid(self):
        """Check whether the configuration is valid."""
        self.F = check_valid(self.F, int, 1)
        self.D = check_valid(self.D, int, 1)
        self.C = check_valid(self.C, int, 1)
        self.S = check_valid(self.S, int, 1)
        self.R = check_valid(self.R, int, 1)
        self.I = check_valid(self.I, int, 1)
        self.dropout = check_valid(self.dropout, float, 0.0, 1.0)
        assert self.leaf_base_class is not None, Exception(
            "RatSpnConfig.leaf_base_class parameter was not set!"
        )
        assert isinstance(self.leaf_base_class, type) and issubclass(
            self.leaf_base_class, Leaf
        ), f"Parameter RatSpnConfig.leaf_base_class must be a subclass type of Leaf but was {self.leaf_base_class}."

        if 2 ** self.D > self.F:
            raise Exception(
                f"The tree depth D={self.D} must be <= {np.floor(np.log2(self.F))} (log2(in_features)."
            )

    def __setattr__(self, key, value):
        """
        Implement __setattr__ so that an EinetConfig object can be created empty `EinetConfig()` and properties can be
        set afterwards.
        """
        if hasattr(self, key):
            super().__setattr__(key, value)
        else:
            raise AttributeError(f"EinetConfig object has no attribute {key}")


class Einet(nn.Module):
    """
    Einet RAT SPN PyTorch implementation with layer-wise tensors.

    See also:
    - RAT SPN: https://arxiv.org/abs/1806.01910
    - EinsumNetworks: https://arxiv.org/abs/2004.06231
    """

    def __init__(self, config: EinetConfig):
        """
        Create a RatSpn based on a configuration object.

        Args:
            config (RatSpnConfig): RatSpn configuration object.
        """
        super().__init__()
        config.assert_valid()
        self.config = config

        # Construct the architecture
        self._build()

        # Initialize weights
        self._init_weights()

    def forward(self, x: torch.Tensor, marginalization_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through RatSpn. Computes the conditional log-likelihood P(X | C).

        Args:
            x: Input.
            marginalization_mask: Leaf marginalization mask. True indicates, that the specific scope
                is missing.

        Returns:
            Log-likelihood tensor of the input: p(X) or p(X | C) if number of classes > 1.
        """
        # Apply leaf distributions (replace marginalization indicators with 0.0 first)
        x = self._leaf(x, marginalization_mask)

        # Pass through intermediate layers
        x = self._forward_layers(x)

        # Merge results from the different repetitions into the channel dimension
        batch_size, features, channels, repetitions = x.size()
        assert features == 1  # number of features should be 1 at this point
        assert channels == self.config.C

        # Treat repetitions as additional channels at this point
        x = x.reshape(batch_size, 1, channels * repetitions, 1)

        # Apply C sum node outputs
        x = self.root(x)

        # Remove repetition dimension
        x = x.squeeze(3)

        # Remove in_features dimension
        x = x.squeeze(1)

        return x

    def _forward_layers(self, x):
        """
        Forward pass through the inner sum and product layers.

        Args:
            x: Input.

        Returns:
            torch.Tensor: Output of the last layer before the root layer.
        """
        # Forward to inner product and sum layers
        for layer in self._inner_layers:
            x = layer(x)
        return x

    def _build(self):
        """Construct the internal architecture of the RatSpn."""
        # Build the SPN bottom up:
        # Definition from RAT Paper
        # Leaf Region:      Create I leaf nodes
        # Root Region:      Create C sum nodes
        # Internal Region:  Create S sum nodes
        # Partition:        Cross products of all child-regions

        # Construct leaf
        self._leaf = self._build_input_distribution()
        self._inner_layers: List[EinsumLayer] = nn.ModuleList()

        # Sum and product layers
        for i in np.arange(start=self.config.D, stop=0, step=-1):
            # Current in_features
            in_features = 2 ** i

            _out_channels = self.config.S if i > 1 else self.config.C
            einsumlayer = EinsumLayer(
                in_features=in_features,
                in_channels=self.config.S,
                out_channels=_out_channels,
                num_repetitions=self.config.R,
                dropout=self.config.dropout,
            )
            self._inner_layers.append(einsumlayer)

        # Construct root layer
        self.root = Sum(
            in_channels=self.config.R * self.config.C,
            in_features=1,
            num_repetitions=1,
            out_channels=self.config.C,
        )

        # Construct sampling root with weights according to priors for sampling
        self._sampling_root = Sum(
            in_channels=self.config.C, in_features=1, out_channels=1, num_repetitions=1
        )
        self._sampling_root.weights = nn.Parameter(
            torch.ones(size=(1, self.config.C, 1, 1)) * torch.tensor(1 / self.config.C),
            requires_grad=False,
        )

    def _build_input_distribution(self):
        """Construct the input distribution layer."""
        # Cardinality is the size of the region in the last partitions
        base_leaf = self.config.leaf_base_class(
            out_channels=self.config.I,
            in_features=self.config.F,
            dropout=self.config.dropout,
            num_repetitions=self.config.R,
            **self.config.leaf_base_kwargs,
        )

        return FactorizedLeaf(
            in_features=self.config.F,
            out_features=2 ** self.config.D,
            num_repetitions=self.config.R,
            base_leaf=base_leaf,
        )

    @property
    def __device(self):
        """Small hack to obtain the current device."""
        return self._sampling_root.weights.device

    def _init_weights(self):
        """Initiale the weights. Calls `_init_weights` on all modules that have this method."""
        for module in self.modules():
            if hasattr(module, "_init_weights") and module != self:
                module._init_weights()
                continue

            if isinstance(module, Sum):
                truncated_normal_(module.weights, std=0.5)
                continue

            if isinstance(module, RatNormal):
                truncated_normal_(module.stds, std=0.1)
                continue

    def mpe(self, evidence: torch.Tensor) -> torch.Tensor:
        """
        Perform MPE given some evidence.

        Args:
            evidence: Input evidence. Must contain some NaN values.
        Returns:
            torch.Tensor: Clone of input tensor with NaNs replaced by MPE estimates.
        """
        return self.sample(evidence=evidence, is_mpe=True)

    def sample(
        self,
        num_samples: int = None,
        class_index=None,
        evidence: torch.Tensor = None,
        is_mpe: bool = False,
        temperature_leaves: float = 1.0,
        temperature_sums: float = 1.0,
        marginalized_scopes: List[int] = None,
    ):
        """
        Sample from the distribution represented by this SPN.

        Possible valid inputs:

        - `n`: Generates `n` samples.
        - `n` and `class_index (int)`: Generates `n` samples from P(X | C = class_index).
        - `class_index (List[int])`: Generates `len(class_index)` samples. Each index `c_i` in `class_index` is mapped
            to a sample from P(X | C = c_i)
        - `evidence`: If evidence is given, samples conditionally and fill NaN values.

        Args:
            n: Number of samples to generate.
            class_index: Class index. Can be either an int in combination with a value for `n` which will result in `n`
                samples from P(X | C = class_index). Or can be a list of ints which will map each index `c_i` in the
                list to a sample from P(X | C = c_i).
            evidence: Evidence that can be provided to condition the samples. If evidence is given, `n` and
                `class_index` must be `None`. Evidence must contain NaN values which will be imputed according to the
                distribution represented by the SPN. The result will contain the evidence and replace all NaNs with the
                sampled values.
            is_mpe: Flag to perform max sampling (MPE).
            temperature_leaves: Variance scaling for leaf distribution samples.
            temperature_leaves: Variance scaling for sum node categorical sampling.

        Returns:
            torch.Tensor: Samples generated according to the distribution specified by the SPN.

        """
        assert (
            class_index is None or evidence is None
        ), "Cannot provide both, evidence and class indices."
        assert (
            num_samples is None or evidence is None
        ), "Cannot provide both, number of samples to generate (n) and evidence."

        # Check if evidence contains nans
        if evidence is not None:
            # Set n to the number of samples in the evidence
            num_samples = evidence.shape[0]

        with provide_evidence(self, evidence, marginalized_scopes):
            # If class is given, use it as base index
            if class_index is not None:
                if isinstance(class_index, list):
                    indices = torch.tensor(class_index, device=self.__device).view(-1, 1)
                    num_samples = indices.shape[0]
                else:
                    indices = torch.empty(size=(num_samples, 1), device=self.__device)
                    indices.fill_(class_index)

                # Create new sampling context
                ctx = SamplingContext(
                    num_samples=num_samples,
                    parent_indices=indices,
                    repetition_indices=None,
                    is_mpe=is_mpe,
                    temperature_leaves=temperature_leaves,
                    temperature_sums=temperature_sums,
                    num_repetitions=self.config.R,
                )
            else:
                # Start sampling one of the C root nodes TODO: check what happens if C=1
                ctx = SamplingContext(
                    num_samples=num_samples,
                    is_mpe=is_mpe,
                    temperature_leaves=temperature_leaves,
                    temperature_sums=temperature_sums,
                    num_repetitions=self.config.R,
                )
                ctx = self._sampling_root.sample(context=ctx)

            # Sample from RatSpn root layer: Results are indices into the stacked output channels of
            # the last layer (num_class) of all repetitions
            ctx.repetition_indices = torch.zeros(num_samples, dtype=int, device=self.__device)
            ctx = self.root.sample(context=ctx)

            # Obtain repetition indices
            ctx.repetition_indices = (ctx.parent_indices % self.config.R).squeeze(1)
            # Shift indices
            ctx.parent_indices = ctx.parent_indices // self.config.R
            # ctx.parent_indices = ctx.parent_indices % num_roots

            # Now each sample in `indices` belongs to one repetition, index in `repetition_indices`

            # Continue at layers
            # Sample inner layers in reverse order (starting from topmost)
            for layer in reversed(self._inner_layers):
                ctx = layer.sample(num_samples=ctx.num_samples, context=ctx)

            # Sample leaf
            samples = self._leaf.sample(context=ctx)

            if evidence is not None:
                # First make a copy such that the original object is not changed
                evidence = evidence.clone()
                evidence[:, marginalized_scopes] = samples[:, marginalized_scopes]
                return evidence
            else:
                return samples


if __name__ == "__main__":
    from distributions import Normal

    torch.manual_seed(0)

    # Input dimensions
    in_features = 8
    batchsize = 5

    # Create input sample
    x = torch.randn(batchsize, in_features)

    # Construct Einet
    config = EinetConfig(
        in_features=in_features,
        D=2,
        S=3,
        I=3,
        R=2,
        dropout=0.0,
        C=1,
        leaf_base_class=RatNormal,
        leaf_base_kwargs=dict(min_sigma=1e-3, max_sigma=1.0),
    )
    einet = Einet(config)

    # Compute log-likelihoods
    lls = einet(x)
    print(f"lls={lls}")
    print(f"lss.shape={lls.shape}")

    # Construct samples
    samples = einet.sample(2)
    print(f"samples={samples}")
    print(f"samples.shape={samples.shape}")

    # Optimize Einet parameters (weights and leaf params)
    optim = torch.optim.Adam(einet.parameters(), lr=0.001)

    for _ in range(1000):
        optim.zero_grad()

        # Forward pass: log-likelihoods
        lls = einet(x)

        # Backprop NLL loss
        nlls = -1 * lls.sum()
        nlls.backward()

        # Update weights
        optim.step()

    # Construct samples
    samples = einet.sample(2)
    print(f"x={x}")
    print(f"samples={samples}")
    print(f"samples.shape={samples.shape}")
