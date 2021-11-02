#!/usr/bin/env python3

from typing import List
import numpy as np
import torch
from torch import nn
from clipper import DistributionClipper

from layers import EinsumLayer, Product, Sum
from utils import SamplingContext, invert_permutation


class Einet(nn.Module):
    def __init__(self, K, D, R, in_features, leaf_cls, C=1):
        """EinsumNetwork Module.

        Args:
            K (int): Number of sum nodes in each layer.
            D (int): Depth.
            R (int): Number of repetitions.
            C (int): Number of root notes (classes, if applicable).
            in_features (int): Number of input features.
            leaf_cls: Leaf layer class.
        """
        super().__init__()
        self.in_features = in_features
        self.in_channels = K
        self.num_sums = K
        self.depth = D
        self.num_repetitions = R
        self.num_classes = C
        self._make_random_repetition_permutation_indices()

        # Create leaf layer
        self.leaf = leaf_cls(in_features=in_features, out_channels=K, num_repetitions=R)

        layers = []
        _in_features = in_features
        _in_channels = self.in_channels
        for d in range(self.depth):
            # Last channel should output only a single sum node
            if d < self.depth - 1:
                out_channels = self.num_sums
            else:
                out_channels = 1

            l = EinsumLayer(
                in_features=_in_features,
                in_channels=_in_channels,
                out_channels=out_channels,
                num_repetitions=R,
            )
            layers.append(l)
            _in_features = l._out_features
            _in_channels = self.num_sums

        self.layers = nn.ModuleList(layers)
        self.prod = Product(_in_features, cardinality=_in_features)
        self.root = Sum(in_channels=R, in_features=1, out_channels=self.num_classes, num_repetitions=1)

    def _make_random_repetition_permutation_indices(self):
        """Create random permutation indices for each repetition."""
        self.rand_indices = torch.empty(size=(self.in_features, self.num_repetitions))
        for r in range(self.num_repetitions):
            # Each repetition has its own randomization
            self.rand_indices[:, r] = torch.tensor(np.random.permutation(self.in_features))

        self.rand_indices = self.rand_indices.long()


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
        x = x.repeat((1, 1, self.num_repetitions))  # Repeat R times

        # Random permutation
        for r in range(self.num_repetitions):
            # Get permutation indices for the r-th repetition
            perm_indices = self.rand_indices[:, r]

            # Permute the features of the r-th version of x using the indices
            x[:, :, r] = x[:, perm_indices, r]

        return x

    def forward(self, x: torch.Tensor):
        # Apply feature randomization for each repetition
        x = self._randomize(x)

        log_p = self.leaf(x)

        # Forward through all layers, bottom up
        for layer in self.layers:
            log_p = layer(log_p)

        # Root layer merges all scopes that are left
        log_p = self.prod(log_p)

        # Shift repetition dimension to build sum over repetitions
        log_p = log_p.permute(0, 1, 3, 2)
        log_p = self.root(log_p)
        log_p = log_p.view(x.shape[0])

        return log_p

    def sample(self, n: int) -> torch.Tensor:
        """Sample from the Einet model.

        Args:
            n (int): Number of samples to generate.

        Returns:
            torch.Tensor: Generated samples.
        """
        context = SamplingContext(num_samples=n)

        context = self.root.sample(context=context)

        # Exchange parent and repetition indices since thie root layer
        # models a mixture over the repetitions
        context = SamplingContext(
            num_samples=context.num_samples,
            parent_indices=context.repetition_indices.unsqueeze(1),
            repetition_indices=context.parent_indices.squeeze(1),
        )

        # Sample from scope merging product layer
        context = self.prod.sample(context.num_samples, context)

        # Sample from all other (EisumLayers) layers in reverse (top-down)
        for layer in reversed(self.layers):
            context = layer.sample(context.num_samples, context)

        # Sample from leaf layer
        samples = self.leaf.sample(context.num_samples, context)

        # Invert permutation
        for i in range(n):
            rep_index = context.repetition_indices[i]
            inv_rand_indices = invert_permutation(self.rand_indices[:, rep_index])
            samples[i, :] = samples[i, inv_rand_indices]

        return samples



import logging
from typing import Dict, Type

import numpy as np
import torch
from dataclasses import dataclass
from torch import nn

from distributions import Leaf
from layers import CrossProduct, Sum
from type_checks import check_valid
from utils import provide_evidence, SamplingContext
from distributions import IndependentMultivariate, RatNormal, truncated_normal_

logger = logging.getLogger(__name__)


def invert_permutation(p: torch.Tensor):
    """
    The argument p is assumed to be some permutation of 0, 1, ..., len(p)-1.
    Returns an array s, where s[i] gives the index of i in p.
    Taken from: https://stackoverflow.com/a/25535723, adapted to PyTorch.
    """
    s = torch.empty(p.shape[0], dtype=p.dtype, device=p.device)
    s[p] = torch.arange(p.shape[0])
    return s


@dataclass
class RatSpnConfig:
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
        assert self.leaf_base_class is not None, Exception("RatSpnConfig.leaf_base_class parameter was not set!")
        assert isinstance(self.leaf_base_class, type) and issubclass(
            self.leaf_base_class, Leaf
        ), f"Parameter RatSpnConfig.leaf_base_class must be a subclass type of Leaf but was {self.leaf_base_class}."

        if 2 ** self.D > self.F:
            raise Exception(f"The tree depth D={self.D} must be <= {np.floor(np.log2(self.F))} (log2(in_features).")

    def __setattr__(self, key, value):
        if hasattr(self, key):
            super().__setattr__(key, value)
        else:
            raise AttributeError(f"RatSpnConfig object has no attribute {key}")


class RatSpn(nn.Module):
    """
    RAT SPN PyTorch implementation with layer-wise tensors.

    See also:
    https://arxiv.org/abs/1806.01910
    """

    def __init__(self, config: RatSpnConfig):
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
        self.rand_indices = torch.empty(size=(self.config.F, self.config.R))
        for r in range(self.config.R):
            # Each repetition has its own randomization
            self.rand_indices[:, r] = torch.tensor(np.random.permutation(self.config.F))

        self.rand_indices = self.rand_indices.long()

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

        # Apply leaf distributions
        x = self._leaf(x)

        # Pass through intermediate layers
        x = self._forward_layers(x)

        # Merge results from the different repetitions into the channel dimension
        N, d, c, r = x.size()
        assert d == 1  # number of features should be 1 at this point
        # x = x.view(n, d, c * r, 1)
        x = x.reshape(N, d, c * r, 1)

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

            einsumlayer = EinsumLayer(in_features=in_features, in_channels=self.config.S, out_channels=self.config.S, num_repetitions=self.config.R)
            self._inner_layers.append(einsumlayer)

        # Construct root layer
        self.root = Sum(
            in_channels=self.config.R * self.config.S, in_features=1, num_repetitions=1, out_channels=self.config.C
        )

        # Construct sampling root with weights according to priors for sampling
        self._sampling_root = Sum(in_channels=self.config.C, in_features=1, out_channels=1, num_repetitions=1)
        self._sampling_root.weights = nn.Parameter(
            torch.ones(size=(1, self.config.C, 1, 1)) * torch.tensor(1 / self.config.C), requires_grad=False
        )

    def _build_input_distribution(self):
        """Construct the input distribution layer."""
        # Cardinality is the size of the region in the last partitions
        cardinality = np.ceil(self.config.F / (2 ** self.config.D)).astype(int)
        return IndependentMultivariate(
            in_features=self.config.F,
            out_channels=self.config.I,
            num_repetitions=self.config.R,
            cardinality=cardinality,
            dropout=self.config.dropout,
            leaf_base_class=self.config.leaf_base_class,
            leaf_base_kwargs=self.config.leaf_base_kwargs,
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

    def sample(self, num_samples: int = None, class_index=None, evidence: torch.Tensor = None, is_mpe: bool = False):
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

        Returns:
            torch.Tensor: Samples generated according to the distribution specified by the SPN.

        """
        assert class_index is None or evidence is None, "Cannot provide both, evidence and class indices."
        assert num_samples is None or evidence is None, "Cannot provide both, number of samples to generate (n) and evidence."

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
                ctx = SamplingContext(num_samples=num_samples, parent_indices=indices, repetition_indices=None, is_mpe=is_mpe)
            else:
                # Start sampling one of the C root nodes TODO: check what happens if C=1
                ctx = SamplingContext(num_samples=num_samples, is_mpe=is_mpe)
                ctx = self._sampling_root.sample(context=ctx)

            # Sample from RatSpn root layer: Results are indices into the stacked output channels of all repetitions
            ctx.repetition_indices = torch.zeros(num_samples, dtype=int, device=self.__device)
            ctx = self.root.sample(context=ctx)

            # Indexes will now point to the stacked channels of all repetitions (R * S^2 (if D > 1)
            # or R * I^2 (else)).
            root_in_channels = self.root.in_channels // self.config.R
            # Obtain repetition indices
            ctx.repetition_indices = (ctx.parent_indices // root_in_channels).squeeze(1)
            # Shift indices
            ctx.parent_indices = ctx.parent_indices % root_in_channels

            # Now each sample in `indices` belongs to one repetition, index in `repetition_indices`

            # Continue at layers
            # Sample inner layers in reverse order (starting from topmost)
            for layer in reversed(self._inner_layers):
                ctx = layer.sample(N = ctx.num_samples, context=ctx)

            # Sample leaf
            samples = self._leaf.sample(context=ctx)

            # Invert permutation
            for i in range(num_samples):
                rep_index = ctx.repetition_indices[i]
                inv_rand_indices = invert_permutation(self.rand_indices[:, rep_index])
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
