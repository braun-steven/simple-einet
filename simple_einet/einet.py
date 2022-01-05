#!/usr/bin/env python3

from collections import defaultdict

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple, Type, Union

import numpy as np
import torch
from torch import nn

# from torchpq.clustering import MinibatchKMeans
from fast_pytorch_kmeans import KMeans

from .distributions import AbstractLeaf, RatNormal, truncated_normal_
from .factorized_leaf_layer import FactorizedLeaf
from .layers import Sum
from .type_checks import OutOfBoundsException, check_valid
from .utils import SamplingContext, provide_evidence
from .einsum_layer import EinsumLayer, EinsumMixingLayer

logger = logging.getLogger(__name__)


@dataclass
class EinetConfig:
    """
    Class for keeping the RatSpn config. Parameter names are according to the original RatSpn paper.

    in_shape: Tuple[int, int]  # Input shape
    depth: int  # Tree depth
    num_sums: int  # Number of sum nodes at each layer
    num_leaves: int  # Number of distributions for each scope at the leaf layer
    num_repetitions: int  # Number of repetitions
    num_classes: int  # Number of root heads / Number of classes
    dropout: float  # Dropout probabilities for leaves and sum layers
    leaf_base_class: Type  # Type of the leaf base class (Normal, Bernoulli, etc)
    leaf_base_kwargs: Dict  # Parameters for the leaf base class
    """

    in_shape: Tuple[int, int] = None
    num_sums: int = None
    num_leaves: int = None
    num_repetitions: int = None
    num_classes: int = None
    depth: int = None
    dropout: float = None
    leaf_type: Type = None
    leaf_kwargs: Dict[str, Any] = None

    @property
    def num_features(self):
        return np.prod(self.in_shape)

    @property
    def in_width(self):
        return self.in_shape[0]

    @property
    def in_height(self):
        return self.in_shape[1]

    def assert_valid(self):
        """Check whether the configuration is valid."""

        # Check that each dimension is valid
        for size in self.in_shape:
            check_valid(size, int, 1)

        self.depth = check_valid(self.depth, int, 1)
        self.num_classes = check_valid(self.num_classes, int, 1)
        self.num_sums = check_valid(self.num_sums, int, 1)
        self.num_repetitions = check_valid(self.num_repetitions, int, 1)
        self.num_leaves = check_valid(self.num_leaves, int, 1)
        self.dropout = check_valid(self.dropout, float, 0.0, 1.0, allow_none=True)
        assert self.leaf_type is not None, Exception(
            "EinetConfig.leaf_base_class parameter was not set!"
        )
        assert isinstance(self.leaf_type, type) and issubclass(
            self.leaf_type, AbstractLeaf
        ), f"Parameter EinetConfig.leaf_base_class must be a subclass type of Leaf but was {self.leaf_type}."

        if 2 ** self.depth > self.num_features:
            raise Exception(
                f"The tree depth D={self.depth} must be <= {np.floor(np.log2(self.num_features))} (log2(in_features))."
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
        x = self.leaf(x, marginalization_mask)

        # Pass through intermediate layers
        x = self._forward_layers(x)

        # Merge results from the different repetitions into the channel dimension
        batch_size, height, width, sums, repetitions = x.size()
        assert width == 1
        assert height == 1
        assert sums == self.config.num_classes

        # Apply C sum node outputs
        x = self.root(x)

        # Remove height/width dimension
        x = x.squeeze(1).squeeze(1)

        # Final shape check
        assert x.shape == (batch_size, self.config.num_classes)

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
        for layer in self.einsum_layers:
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


        einsum_layers = []

        # Start first layer with width split (therefore, the in_shape has to be [2, 1])
        in_shape = [2, 1]
        split_dim = "h"
        for i in np.arange(start=1, stop=self.config.depth + 1):

            _num_sums_in = self.config.num_sums if i < self.config.depth else self.config.num_leaves
            # _num_sums_out = self.config.num_sums if i > 1 else self.config.num_classes
            _num_sums_out = self.config.num_sums if i > 1 else self.config.num_classes
            layer = EinsumLayer(
                in_shape=in_shape,
                num_sums_in=_num_sums_in,
                num_sums_out=_num_sums_out,
                num_repetitions=self.config.num_repetitions,
                split_dim=split_dim
            )
            einsum_layers.append(layer)

            # Alternate between splits
            if split_dim == "h":
                split_dim = "w"
                in_shape = [in_shape[0], in_shape[1] * 2]
            else:
                split_dim = "h"
                in_shape = [in_shape[0] * 2, in_shape[1]]




        # Construct leaf
        self.leaf = self._build_input_distribution(out_shape=einsum_layers[-1].in_shape)

        # List layers in a bottom-to-top fashion
        self.einsum_layers: Sequence[EinsumLayer] = nn.ModuleList(reversed(einsum_layers))

        # Construct root layer which mixes the repetitions
        self.root = EinsumMixingLayer(
            in_shape=[1, 1],
            num_sums_in=self.config.num_repetitions,
            num_sums_out=self.config.num_classes,
        )

        # Construct sampling root with weights according to priors for sampling
        self._sampling_root = Sum(
            num_sums_in=self.config.num_classes,
            in_shape=[
                1, 1
            ],
            num_sums_out=1,
            num_repetitions=1,
        )
        self._sampling_root.weights = nn.Parameter(
            torch.ones(size=(1, 1, self.config.num_classes, 1, 1))
            * torch.tensor(1 / self.config.num_classes),
            requires_grad=False,
        )

    def _build_input_distribution(self, out_shape: Tuple[int, int]):
        """Construct the input distribution layer."""
        # Cardinality is the size of the region in the last partitions
        base_leaf = self.config.leaf_type(
            in_shape=self.config.in_shape,
            num_leaves=self.config.num_leaves,
            num_repetitions=self.config.num_repetitions,
            **self.config.leaf_kwargs,
        )

        return FactorizedLeaf(
            in_shape=base_leaf.out_shape,
            out_shape=out_shape,
            num_repetitions=self.config.num_repetitions,
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

    def mpe(
        self,
        evidence: torch.Tensor = None,
        marginalized_scopes: List[int] = None,
    ) -> torch.Tensor:
        """
        Perform MPE given some evidence.

        Args:
            evidence: Input evidence. Must contain some NaN values.
        Returns:
            torch.Tensor: Clone of input tensor with NaNs replaced by MPE estimates.
        """
        return self.sample(evidence=evidence, is_mpe=True, marginalized_scopes=marginalized_scopes)

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

        - `num_samples`: Generates `num_samples` samples.
        - `num_samples` and `class_index (int)`: Generates `num_samples` samples from P(X | C = class_index).
        - `class_index (List[int])`: Generates `len(class_index)` samples. Each index `c_i` in `class_index` is mapped
            to a sample from P(X | C = c_i)
        - `evidence`: If evidence is given, samples conditionally and fill NaN values.

        Args:
            num_samples: Number of samples to generate.
            class_index: Class index. Can be either an int in combination with a value for `num_samples` which will result in `num_samples`
                samples from P(X | C = class_index). Or can be a list of ints which will map each index `c_i` in the
                list to a sample from P(X | C = c_i).
            evidence: Evidence that can be provided to condition the samples. If evidence is given, `num_samples` and
                `class_index` must be `None`. Evidence must contain NaN values which will be imputed according to the
                distribution represented by the SPN. The result will contain the evidence and replace all NaNs with the
                sampled values.
            is_mpe: Flag to perform max sampling (MPE).
            temperature_leaves: Variance scaling for leaf distribution samples.
            temperature_sums: Variance scaling for sum node categorical sampling.

        Returns:
            torch.Tensor: Samples generated according to the distribution specified by the SPN.

        """
        assert (
            class_index is None or evidence is None
        ), "Cannot provide both, evidence and class indices."
        assert (
            num_samples is None or evidence is None
        ), "Cannot provide both, number of samples to generate (num_samples) and evidence."

        # Check if evidence contains nans
        if evidence is not None:
            # Set n to the number of samples in the evidence
            num_samples = evidence.shape[0]
        elif num_samples is None:
            num_samples = 1

        with provide_evidence(self, evidence, marginalized_scopes):
            # If class is given, use it as base index
            if class_index is not None:
                if isinstance(class_index, list):
                    indices = torch.tensor(class_index, device=self.__device).view(-1, 1, 1)
                    num_samples = indices.shape[0]
                else:
                    indices = torch.empty(size=(num_samples, 1, 1), device=self.__device)
                    indices.fill_(class_index)

                # Create new sampling context
                ctx = SamplingContext(
                    num_samples=num_samples,
                    indices_out=indices,
                    indices_repetition=None,
                    is_mpe=is_mpe,
                    temperature_leaves=temperature_leaves,
                    temperature_sums=temperature_sums,
                    num_repetitions=self.config.num_repetitions,
                )
            else:
                # Start sampling one of the C root nodes TODO: check what happens if C=1
                ctx = SamplingContext(
                    num_samples=num_samples,
                    is_mpe=is_mpe,
                    temperature_leaves=temperature_leaves,
                    temperature_sums=temperature_sums,
                    num_repetitions=self.config.num_repetitions,
                )
                ctx = self._sampling_root.sample(context=ctx)

            # Sample from RatSpn root layer: Results are indices into the stacked output channels of
            # the last layer (num_class) of all repetitions

            # Save parent indices that were sampled from the sampling root
            indices_out_pre_root = ctx.indices_out

            ctx.indices_repetition = torch.zeros(num_samples, dtype=int, device=self.__device)
            ctx = self.root.sample(context=ctx)

            # Obtain repetition indices
            ctx.indices_repetition = ctx.indices_out.view(num_samples)
            ctx.indices_out = indices_out_pre_root

            # Sample inner layers in reverse order (starting from topmost)
            for layer in reversed(self.einsum_layers):
                ctx = layer.sample(num_samples=ctx.num_samples, context=ctx)

            # Sample leaf
            samples = self.leaf.sample(context=ctx)

            if evidence is not None:
                # First make a copy such that the original object is not changed
                evidence = evidence.clone()
                evidence[:, marginalized_scopes] = samples[:, marginalized_scopes]
                return evidence
            else:
                return samples


class EinetMixture(nn.Module):
    def __init__(self, n_components: int, einet_config: EinetConfig):
        super().__init__()
        self.n_components = check_valid(n_components, expected_type=int, lower_bound=1)
        self.config = einet_config

        einets = []

        for i in range(n_components):
            einets.append(Einet(einet_config))

        self.einets: Sequence[Einet] = nn.ModuleList(einets)
        self._kmeans = KMeans(n_clusters=self.n_components, mode="euclidean", verbose=1)
        self.mixture_weights = nn.Parameter(torch.empty(n_components), requires_grad=False)
        self.centroids = nn.Parameter(
            torch.empty(n_components, einet_config.num_features), requires_grad=False
        )

    @torch.no_grad()
    def initialize(self, data: torch.Tensor):
        data = data.float()  # input has to be [d, n]
        self._kmeans.fit(data)

        self.mixture_weights.data = (
            self._kmeans.num_points_in_clusters / self._kmeans.num_points_in_clusters.sum()
        )
        self.centroids.data = self._kmeans.centroids

    def _predict_cluster(self, x, marginalized_scopes: List[int] = None):
        if marginalized_scopes is not None:
            keep_idx = list(
                sorted([i for i in range(self.config.num_features) if i not in marginalized_scopes])
            )
            centroids = self.centroids[:, keep_idx]
            x = x[:, keep_idx]
        else:
            centroids = self.centroids
        return self._kmeans.max_sim(a=x.float(), b=centroids)[1]

    def _separate_data_by_cluster(self, x: torch.Tensor, marginalized_scope: List[int]):
        cluster_idxs = self._predict_cluster(x, marginalized_scope).tolist()

        separated_data = defaultdict(list)
        separated_idxs = defaultdict(list)
        for data_idx, cluster_idx in enumerate(cluster_idxs):
            separated_data[cluster_idx].append(x[data_idx])
            separated_idxs[cluster_idx].append(data_idx)

        return separated_idxs, separated_data

    def forward(self, x, marginalized_scope: torch.Tensor = None):
        assert self._kmeans is not None, "EinetMixture has not been initialized yet."

        separated_idxs, separated_data = self._separate_data_by_cluster(x, marginalized_scope)

        lls_result = []
        data_idxs_all = []
        for cluster_idx, data_list in separated_data.items():
            data_tensor = torch.stack(data_list, dim=0)
            lls = self.einets[cluster_idx](data_tensor)

            data_idxs = separated_idxs[cluster_idx]
            for data_idx, ll in zip(data_idxs, lls):
                lls_result.append(ll)
                data_idxs_all.append(data_idx)

        # Sort results into original order as observed in the batch
        L = [(data_idxs_all[i], i) for i in range(len(data_idxs_all))]
        L.sort()
        _, permutation = zip(*L)
        permutation = torch.tensor(permutation, device=x.device).view(-1)
        lls_result = torch.stack(lls_result)
        lls_sorted = lls_result[permutation]

        return lls_sorted

    def sample(
        self,
        num_samples: int = None,
        num_samples_per_cluster: int = None,
        class_index=None,
        evidence: torch.Tensor = None,
        is_mpe: bool = False,
        temperature_leaves: float = 1.0,
        temperature_sums: float = 1.0,
        marginalized_scopes: List[int] = None,
    ):
        assert num_samples is None or num_samples_per_cluster is None
        if num_samples is None and num_samples_per_cluster is not None:
            num_samples = num_samples_per_cluster * self.n_components

        # Check if evidence contains nans
        if evidence is not None:
            # Set n to the number of samples in the evidence
            num_samples = evidence.shape[0]
        elif num_samples is None:
            num_samples = 1

        if is_mpe:
            # Take cluster idx with largest weights
            cluster_idxs = [self.mixture_weights.argmax().item()]
        else:
            if num_samples_per_cluster is not None:
                cluster_idxs = (
                    torch.arange(self.n_components)
                    .repeat_interleave(num_samples_per_cluster)
                    .tolist()
                )
            else:
                # Sample from categorical over weights
                cluster_idxs = (
                    torch.distributions.Categorical(probs=self.mixture_weights)
                    .sample((num_samples,))
                    .tolist()
                )

        if evidence is None:
            # Sample without evidence
            separated_idxs = defaultdict(int)
            for cluster_idx in cluster_idxs:
                separated_idxs[cluster_idx] += 1

            samples_all = []
            for cluster_idx, num_samples_cluster in separated_idxs.items():
                samples = self.einets[cluster_idx].sample(
                    num_samples_cluster,
                    class_index=class_index,
                    evidence=evidence,
                    is_mpe=is_mpe,
                    temperature_leaves=temperature_leaves,
                    temperature_sums=temperature_sums,
                    marginalized_scopes=marginalized_scopes,
                )
                samples_all.append(samples)

            samples = torch.cat(samples_all, dim=0)
        else:
            # Sample with evidence
            separated_idxs, separated_data = self._separate_data_by_cluster(
                evidence, marginalized_scopes
            )

            samples_all = []
            evidence_idxs_all = []
            for cluster_idx, evidence_pre_cluster in separated_data.items():
                evidence_per_cluster = torch.stack(evidence_pre_cluster, dim=0)
                samples = self.einets[cluster_idx].sample(
                    evidence=evidence_per_cluster,
                    is_mpe=is_mpe,
                    temperature_leaves=temperature_leaves,
                    temperature_sums=temperature_sums,
                    marginalized_scopes=marginalized_scopes,
                )

                evidence_idxs = separated_idxs[cluster_idx]
                for evidence_idx, sample in zip(evidence_idxs, samples):
                    samples_all.append(sample)
                    evidence_idxs_all.append(evidence_idx)

            # Sort results into original order as observed in the batch
            L = [(evidence_idxs_all[i], i) for i in range(len(evidence_idxs_all))]
            L.sort()
            _, permutation = zip(*L)
            permutation = torch.tensor(permutation, device=evidence.device).view(-1)
            samples_all = torch.stack(samples_all)
            samples_sorted = samples_all[permutation]
            samples = samples_sorted

        return samples

    def mpe(
        self,
        evidence: torch.Tensor = None,
        marginalized_scopes: List[int] = None,
    ) -> torch.Tensor:
        """
        Perform MPE given some evidence.

        Args:
            evidence: Input evidence. Must contain some NaN values.
        Returns:
            torch.Tensor: Clone of input tensor with NaNs replaced by MPE estimates.
        """
        return self.sample(evidence=evidence, is_mpe=True, marginalized_scopes=marginalized_scopes)


if __name__ == "__main__":
    torch.manual_seed(0)

    # Input dimensions
    in_features = 8
    batchsize = 5

    # Create input sample
    x = torch.randn(batchsize, in_features)

    # Construct Einet
    config = EinetConfig(
        num_features=in_features,
        depth=2,
        num_sums=3,
        num_leaves=3,
        num_repetitions=2,
        dropout=0.0,
        num_classes=1,
        leaf_type=RatNormal,
        leaf_kwargs=dict(min_sigma=1e-3, max_sigma=1.0),
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
