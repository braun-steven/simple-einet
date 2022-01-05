import warnings
from typing import List, Sequence, Tuple
from torch import nn
import torch.nn.functional as F
import torch
import numpy as np

from .utils import SamplingContext
from .layers import AbstractLayer
from .distributions import AbstractLeaf

# Construct mapping of scopes from in_features -> out_features
def _make_scopes(in_size, out_size, num_repetitions):
    # Number of input scopes that get merged into a single output scope
    cardinality = int(np.round(in_size / out_size))

    # Scope tensor maps a set of input scopes to a single output scope
    scopes = torch.zeros(in_size, out_size, num_repetitions)

    for r in range(num_repetitions):

        # Generate list of all possible input scopes
        # TODO: for RAT use randperm
        # idxs = torch.randperm(n=in_size)
        idxs = torch.arange(in_size)

        # For each output scope: generate corresponding input scopes via a sliding window approach
        for out_idx in range(out_size):

            # Sliding window over input indices
            low = out_idx * cardinality
            high = (out_idx + 1) * cardinality

            # Merge the following input scopes into a single output scope at (out_idx)
            in_idx = idxs[low:high]
            scopes[in_idx, out_idx, r] = 1

    return scopes


class FactorizedLeaf(AbstractLayer):
    """
    A 'meta'-leaf layer that combines multiple scopes of a base-leaf layer via naive factorization.

    Attributes:
        base_leaf: Base leaf layer that contains the actual leaf distribution.
        in_features: Number of input features/RVs.
        out_features: Number of output features/RVs. This determines the factorization group size (round(in_features / out_features))
        scopes: One-hot mapping from which in_features correspond to which out_features.

    """

    def __init__(
        self,
        in_shape: Tuple[int, int],
        out_shape: Tuple[int, int],
        num_repetitions,
        base_leaf: AbstractLeaf,
    ):
        """
        Args:
            in_features (int): Number of input features/RVs.
            out_features (int): Number of output features/RVs.
            num_repetitions (int): Number of repetitions.
            base_leaf (Leaf): Base leaf distribution object.
        """

        super().__init__(in_shape, num_repetitions=num_repetitions)

        self.base_leaf = base_leaf
        self.out_shape = out_shape

        # Separate scopes matrices for height and width
        self.register_buffer("scopes_h", _make_scopes(in_shape[0], out_shape[0], num_repetitions))
        self.register_buffer("scopes_w", _make_scopes(in_shape[1], out_shape[1], num_repetitions))

    def forward(self, x: torch.Tensor, marginalized_scopes: List[int]):
        # Forward through base leaf
        x = self.base_leaf(x, marginalized_scopes)  # [n, c, h, w, num_leaves, r]

        # Factorize input channels
        x = x.sum(dim=1)

        N, H, W, OC, R = x.shape
        assert (H, W) == self.in_shape

        # Merge scopes by naive factorization
        x = torch.einsum("bhwcr,hor->bowcr", x, self.scopes_h)
        x = torch.einsum("bhwcr,wor->bhocr", x, self.scopes_w)

        assert (N, *self.out_shape, OC, R) == x.shape
        return x

    def sample(self, num_samples: int = None, context: SamplingContext = None) -> torch.Tensor:
        # Save original indices_out and set context indices_out to none, such that the out_channel
        # are not filtered in the base_leaf sampling procedure
        indices_out = context.indices_out
        context.indices_out = None
        samples = self.base_leaf.sample(context=context)

        # Check that shapes match as expected
        if samples.dim() == 4:
            assert samples.shape == (
                context.num_samples,
                self.in_shape[0],
                self.in_shape[1],
                self.base_leaf.num_leaves,
            )
        elif samples.dim() == 5:
            assert self.in_shape[0] == samples.shape[1]
            assert self.in_shape[1] == samples.shape[2]
            assert hasattr(self.base_leaf, "cardinality")
            assert samples.shape == (
                context.num_samples,
                self.base_leaf.out_shape[0],
                self.base_leaf.out_shape[1],
                self.base_leaf.cardinality,
                self.base_leaf.num_leaves,
            )

        # Collect final samples in temporary tensor
        if hasattr(self.base_leaf, "cardinality"):
            cardinality = self.base_leaf.cardinality
        else:
            cardinality = 1
        tmp = torch.zeros(
            context.num_samples,
            self.in_shape[0],
            self.in_shape[1],
            device=samples.device,
            dtype=samples.dtype,
        )
        for sample_idx in range(context.num_samples):
            # Get repetition for current sample
            rep = context.indices_repetition[sample_idx]

            # Get out index for current sample
            # Interpretation: For each scope i,j in indices_out, the element (index) at position i,j
            # is the index into the leaf sample output channel (0, ..., num_leaves - 1)
            indices_out_i = indices_out[sample_idx]
            assert list(indices_out_i.shape) == self.out_shape

            # Get scope for the current repetition
            upsize = self.scopes_w.shape[0] // self.scopes_w.shape[1]
            upsample = nn.ConvTranspose2d(1, 1, upsize, stride=upsize)
            upsample.weight[:] = 1.0
            up = upsample(indices_out_i.view(1, 1, *indices_out_i.shape).float())
            up = up.round().long()[0, 0]
            scope_h = self.scopes_h[:, :, rep]  # Which height scopes get merged
            scope_w = self.scopes_w[:, :, rep]  # Which width scopes get merged

            # Turn one-hot encoded in-feature -> out-feature mapping into a linear index
            rnge_in_h = torch.arange(self.out_shape[0], device=samples.device)
            rnge_in_w = torch.arange(self.out_shape[1], device=samples.device)

            # Mapping from in-scope to out-scope
            # Read: element i in the following list is an index j, where
            # i is an index into the in_shape and j is the corresponding index into the out_shape
            scope_h = (scope_h * rnge_in_h).sum(-1).long()
            scope_w = (scope_w * rnge_in_w).sum(-1).long()
            assert scope_h.shape[0] == self.in_shape[0]
            assert scope_w.shape[0] == self.in_shape[1]


            # # Map indices_out from original "out_shape" view to "in_shape" view
            scope_h = scope_h.view(-1, 1).expand(-1, indices_out_i.shape[0])
            scope_w = scope_w.view(1, -1).expand(self.in_shape[1], -1)
            indices_in = indices_out_i.gather(dim=0, index=scope_h)
            indices_in = indices_in.gather(dim=1, index=scope_w)
            # assert (indices_in == up).all()
            indices_in.fill_(sample_idx % samples.shape[-1])
            warnings.warn("Sampling indices fixed in factorizedleaf")


            # TODO: This is not yet working - something is off with the indexing
            # IDEA: maybe linearize scope from the beginning?
            sample_i = samples[sample_idx]
            sample_i_selected = sample_i.gather(dim=2, index=indices_in.unsqueeze(-1)).squeeze(-1)
            tmp[sample_idx] = sample_i_selected
        samples = tmp
        return samples

    def __repr__(self):
        return f"FactorizedLeaf(in_shape={self.in_shape}, out_shape={self.out_shape})"
