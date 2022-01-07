from typing import List
import torch
import numpy as np

from .utils import SamplingContext
from .layers import AbstractLayer
from .distributions import Leaf


class FactorizedLeaf(AbstractLayer):
    """
    A 'meta'-leaf layer that combines multiple scopes of a base-leaf layer via naive factorization.

    Attributes:
        base_leaf: Base leaf layer that contains the actual leaf distribution.
        in_features: Number of input features/RVs.
        out_features: Number of output features/RVs. This determines the factorization group size (round(in_features / out_features))
        scopes: One-hot mapping from which in_features correspond to which out_features.

    """

    def __init__(self, in_features: int, out_features: int, num_repetitions, base_leaf: Leaf):
        """
        Args:
            in_features (int): Number of input features/RVs.
            out_features (int): Number of output features/RVs.
            num_repetitions (int): Number of repetitions.
            base_leaf (Leaf): Base leaf distribution object.
        """

        super().__init__(in_features, num_repetitions=num_repetitions)

        self.base_leaf = base_leaf
        self.out_features = out_features

        # Size of the factorized groups of RVs
        cardinality = int(np.round(self.in_features / self.out_features))

        # Construct mapping of scopes from in_features -> out_features
        scopes = torch.zeros(in_features, self.out_features, num_repetitions)
        for r in range(num_repetitions):
            idxs = torch.randperm(n=self.in_features)
            for o in range(out_features):
                low = o * cardinality
                high = (o + 1) * cardinality
                if o == out_features - 1:
                    high = self.in_features
                scopes[idxs[low:high], o, r] = 1

        self.register_buffer("scopes", scopes)

    def forward(self, x: torch.Tensor, marginalized_scopes: List[int]):
        # Forward through base leaf
        x = self.base_leaf(x, marginalized_scopes)

        # Merge scopes by naive factorization
        x = torch.einsum("bicr,ior->bocr", x, self.scopes)
        return x

    def sample(self, num_samples: int = None, context: SamplingContext = None) -> torch.Tensor:
        # Save original parent_indices and set context parent_indices to none, such that the out_channel
        # are not filtered in the base_leaf sampling procedure
        parent_indices = context.parent_indices
        context.parent_indices = None
        samples = self.base_leaf.sample(context=context)

        # Check that shapes match as expected
        if samples.dim() == 3:
            assert samples.shape == (
                context.num_samples,
                self.in_features,
                self.base_leaf.out_channels,
            )
        elif samples.dim() == 4:
            assert self.in_features == samples.shape[1]
            assert hasattr(self.base_leaf, "cardinality")
            assert samples.shape == (
                context.num_samples,
                self.base_leaf.out_features,
                self.base_leaf.cardinality,
                self.base_leaf.out_channels,
            )

        # Collect final samples in temporary tensor
        if hasattr(self.base_leaf, "cardinality"):
            cardinality = self.base_leaf.cardinality
        else:
            cardinality = 1
        tmp = torch.zeros(
            context.num_samples,
            self.in_features,
            cardinality,
            device=samples.device,
            dtype=samples.dtype,
        )
        for sample_idx in range(context.num_samples):
            # Get correct repetition
            r = context.repetition_indices[sample_idx]

            # Get correct parent_indices
            parent_indices_out = parent_indices[sample_idx]

            # Get scope for the current repetition
            scope = self.scopes[:, :, r]

            # Turn one-hot encoded in-feature -> out-feature mapping into a linear index
            rnge_in = torch.arange(self.out_features, device=samples.device)
            scope = (scope * rnge_in).sum(-1).long()

            # Map parent_indices from original "out_features" view to "in_feautres" view
            parent_indices_in = parent_indices_out[scope]

            # Access base leaf samples based on
            rnge_out = torch.arange(self.in_features, device=samples.device)

            tmp[sample_idx] = samples[sample_idx, rnge_out, ..., parent_indices_in].view(
                self.in_features, cardinality
            )

        samples = tmp.view(context.num_samples, -1)
        return samples

    def __repr__(self):
        return f"FactorizedLeaf(in_features={self.in_features}, out_features={self.out_features})"
