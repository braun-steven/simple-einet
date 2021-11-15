#!/usr/bin/env python3

import logging
from dataclasses import dataclass
from typing import Dict, Type

import numpy as np
import torch
from torch import nn

from clipper import DistributionClipper
from distributions import IndependentMultivariate, Leaf, MultivariateNormal, truncated_normal_
from layers import CrossProduct, EinsumLayer, Product, Sum
from type_checks import check_valid
from utils import SamplingContext, invert_permutation, provide_evidence

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

        # Obtain permutation indices
        self._make_random_repetition_permutation_indices()

    def _make_random_repetition_permutation_indices(self):
        """Create random permutation indices for each repetition."""
        rand_indices = torch.empty(size=(self.config.F, self.config.R))
        for r in range(self.config.R):
            # Each repetition has its own randomization
            rand_indices[:, r] = torch.tensor(np.random.permutation(self.config.F))
        rand_indices = rand_indices.long()

        # Construct inverse indices necessary during sampling
        inv_rand_indices = torch.empty_like(rand_indices)
        for r in range(self.config.R):
            inv_rand_indices[:, r] = invert_permutation(rand_indices[:, r])

        # Register as buffer so it persists when storing etc.
        self.register_buffer("rand_indices", rand_indices)
        self.register_buffer("inv_rand_indices", inv_rand_indices)

    def _randomize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Randomize the input at each repetition according to `self.rand_indices`.

        Args:
            x: Input.

        Returns:
            torch.Tensor: Randomized input along feature axis. Each repetition has its own permutation.
        """
        # Expand input to the number of repetitions
        x = x.unsqueeze(2)  # Make space for repetition axis
        x = x.repeat((1, 1, self.config.R))  # Repeat R times

        # Random permutation
        for r in range(self.config.R):
            # Get permutation indices for the r-th repetition
            perm_indices = self.rand_indices[:, r]

            # Permute the features of the r-th version of x using the indices
            x[:, :, r] = x[:, perm_indices, r]

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through RatSpn. Computes the conditional log-likelihood P(X | C).

        Args:
            x: Input.

        Returns:
            torch.Tensor: Conditional log-likelihood P(X | C) of the input.
        """
        # Apply feature randomization for each repetition
        x = self._randomize(x)

        # Apply leaf distributions (replace marginalization indicators with 0.0 first)
        x = self._leaf(x)

        # Pass through intermediate layers
        x = self._forward_layers(x)

        # Merge results from the different repetitions into the channel dimension
        batch_size, features, channels, repetitions = x.size()
        assert features == 1  # number of features should be 1 at this point
        assert channels == 1  # number of channels should be 1 at this point
        x = x.view(batch_size, 1, repetitions, 1)

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
        self._inner_layers = nn.ModuleList()

        # Sum and product layers
        for i in np.arange(start=self.config.D, stop=0, step=-1):
            # Current in_features
            in_features = 2 ** i

            _out_channels = self.config.S if i > 1 else 1
            einsumlayer = EinsumLayer(
                in_features=in_features,
                in_channels=self.config.S,
                out_channels=_out_channels,
                num_repetitions=self.config.R,
            )
            self._inner_layers.append(einsumlayer)

        # Construct root layer
        self.root = Sum(
            in_channels=self.config.R, in_features=1, num_repetitions=1, out_channels=self.config.C
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
        cardinality = np.ceil(self.config.F / (2 ** self.config.D)).astype(int)
        pad = int(2 ** self.config.D - self.config.F / cardinality)
        return IndependentMultivariate(
            in_features=self.config.F,
            out_channels=self.config.I,
            num_repetitions=self.config.R,
            cardinality=cardinality,
            dropout=self.config.dropout,
            leaf_base_class=self.config.leaf_base_class,
            leaf_base_kwargs=self.config.leaf_base_kwargs,
            pad=pad,
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
            assert (evidence != evidence).any(), "Evidence has no NaN values."

            # Set n to the number of samples in the evidence
            num_samples = evidence.shape[0]

        with provide_evidence(self, evidence):  # May be None but that's ok
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

            # Sample from RatSpn root layer: Results are indices into the stacked output channels of all repetitions
            ctx.repetition_indices = torch.zeros(num_samples, dtype=int, device=self.__device)
            ctx = self.root.sample(context=ctx)

            # Obtain repetition indices
            ctx.repetition_indices, ctx.parent_indices = (
                ctx.parent_indices.squeeze(1),
                ctx.repetition_indices,
            )

            # Now each sample in `indices` belongs to one repetition, index in `repetition_indices`

            # Continue at layers
            # Sample inner layers in reverse order (starting from topmost)
            for layer in reversed(self._inner_layers):
                ctx = layer.sample(num_samples=ctx.num_samples, context=ctx)

            # Sample leaf
            samples = self._leaf.sample(context=ctx)

            # Invert permutation
            for i in range(num_samples):
                rep_index = ctx.repetition_indices[i]
                inv_rand_indices = self.inv_rand_indices[:, rep_index]
                samples[i, :] = samples[i, inv_rand_indices]

            if evidence is not None:
                # Update NaN entries in evidence with the sampled values
                nan_indices = torch.isnan(evidence)

                # First make a copy such that the original object is not changed
                evidence = evidence.clone()
                evidence[nan_indices] = samples[nan_indices]
                return evidence
            else:
                return samples


if __name__ == "__main__":
    from distributions import Normal

    torch.manual_seed(0)

    # Input dimensions
    in_features = 4
    batchsize = 5

    # Create input sample
    x = torch.randn(batchsize, in_features)

    # Construct Einet
    einet = Einet(K=2, D=2, R=2, in_features=in_features, leaf_cls=Normal)

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
    clipper = DistributionClipper()

    for _ in range(1000):
        optim.zero_grad()

        # Forward pass: log-likelihoods
        lls = einet(x)

        # Backprop NLL loss
        nlls = -1 * lls.sum()
        nlls.backward()

        # Update weights
        optim.step()

        # Clip leaf distribution parameters (e.g. std > 0.0, etc.)
        clipper(einet.leaf)

    # Construct samples
    samples = einet.sample(2)
    print(f"x={x}")
    print(f"samples={samples}")
    print(f"samples.shape={samples.shape}")
