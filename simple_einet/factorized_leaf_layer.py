from typing import List
import torch
import numpy as np

from .utils import SamplingContext
from .layers import AbstractLayer
from .distributions import AbstractLeaf


class FactorizedLeaf(AbstractLayer):
    """
    A 'meta'-leaf layer that combines multiple scopes of a base-leaf layer via naive factorization.

    Attributes:
        base_leaf: Base leaf layer that contains the actual leaf distribution.
        in_features: Number of input features/RVs.
        out_features: Number of output features/RVs. This determines the factorization group size (round(in_features / out_features))
        scopes: One-hot mapping from which in_features correspond to which out_features.

    """

    def __init__(self, num_features: int, num_features_out: int, num_repetitions, base_leaf: AbstractLeaf):
        """
        Args:
            in_features (int): Number of input features/RVs.
            out_features (int): Number of output features/RVs.
            num_repetitions (int): Number of repetitions.
            base_leaf (Leaf): Base leaf distribution object.
        """

        super().__init__(num_features, num_repetitions=num_repetitions)

        self.base_leaf = base_leaf
        self.num_features_out = num_features_out

        # Size of the factorized groups of RVs
        cardinality = int(np.round(self.num_features / self.num_features_out))

        # Construct mapping of scopes from in_features -> out_features
        scopes = torch.zeros(num_features, self.num_features_out, num_repetitions)
        for r in range(num_repetitions):
            idxs = torch.randperm(n=self.num_features)
            for o in range(num_features_out):
                low = o * cardinality
                high = (o + 1) * cardinality
                if o == num_features_out - 1:
                    high = self.num_features
                scopes[idxs[low:high], o, r] = 1

        self.register_buffer("scopes", scopes)

    def forward(self, x: torch.Tensor, marginalized_scopes: List[int]):
        # Forward through base leaf
        x = self.base_leaf(x, marginalized_scopes)

        # Factorize input channels
        x = x.sum(dim=1)

        # Merge scopes by naive factorization
        x = torch.einsum("bicr,ior->bocr", x, self.scopes)

        assert x.shape == (x.shape[0], self.num_features_out, self.base_leaf.num_leaves, self.num_repetitions)
        return x

    def sample(self, num_samples: int = None, context: SamplingContext = None) -> torch.Tensor:
        # Save original indices_out and set context indices_out to none, such that the out_channel
        # are not filtered in the base_leaf sampling procedure
        indices_out = context.indices_out
        context.indices_out = None
        samples = self.base_leaf.sample(context=context)
        num_samples = samples.shape[0]

        # Check that shapes match as expected
        if samples.dim() == 4:
            assert samples.shape == (
                context.num_samples,
                self.base_leaf.num_channels,
                self.num_features,
                self.base_leaf.num_leaves,
            )
        elif samples.dim() == 5:
            assert self.num_features == samples.shape[1]
            assert hasattr(self.base_leaf, "cardinality")
            assert samples.shape == (
                context.num_samples,
                self.base_leaf.out_features,
                self.base_leaf.cardinality,
                self.base_leaf.num_leaves,
            )

        # indices_in_tmp = torch.zeros(num_samples, self.num_features, device=samples.device)

        # # Collect final samples in temporary tensor
        # if hasattr(self.base_leaf, "cardinality"):
        #     cardinality = self.base_leaf.cardinality
        # else:
        #     cardinality = 1
        # tmp = torch.zeros(
        #     context.num_samples,
        #     self.base_leaf.num_channels,
        #     self.num_features,
        #     cardinality,
        #     device=samples.device,
        #     dtype=samples.dtype,
        # )
        # for sample_idx in range(context.num_samples):
        #     # Get correct repetition
        #     r = context.indices_repetition[sample_idx]

        #     # Get correct indices_out
        #     indices_out_out = indices_out[sample_idx]

        #     # Get scope for the current repetition
        #     scope = self.scopes[:, :, r]

        #     # Turn one-hot encoded in-feature -> out-feature mapping into a linear index
        #     rnge_in = torch.arange(self.num_features_out, device=samples.device)
        #     scope = (scope * rnge_in).sum(-1).long()

        #     # Map indices_out from original "out_features" view to "in_feautres" view
        #     indices_out_in = indices_out_out[scope]
        #     indices_in_tmp[sample_idx] = indices_out_in

        #     # TODO: Alternative with gather which probably works over all samples
        #     # scope_h = scope_h.view(-1, 1).expand(-1, indices_out_i.shape[0])
        #     # scope_w = scope_w.view(1, -1).expand(self.in_shape[1], -1)
        #     # indices_in = indices_out_i.gather(dim=0, index=scope_h)

        #     # Access base leaf samples based on
        #     rnge_out = torch.arange(self.num_features, device=samples.device)

        #     sample_i = samples[sample_idx, :, rnge_out, ..., indices_out_in].view(
        #         self.num_features, cardinality
        #     )
        #     tmp[sample_idx] = sample_i
        # samples = tmp.view(context.num_samples, -1)

        scopes = self.scopes[..., context.indices_repetition].permute(2, 0, 1)
        rnge_in = torch.arange(self.num_features_out, device=samples.device)
        scopes = (scopes * rnge_in).sum(-1).long()
        indices_in_gather = indices_out.gather(dim=1, index=scopes)

        # assert (indices_in_tmp == indices_in_gather).all()

        indices_in_gather = indices_in_gather.view(num_samples, 1, -1, 1).expand(-1, samples.shape[1], -1, -1)
        samples = samples.gather(dim=-1, index=indices_in_gather)
        samples.squeeze_(-1)  # Remove num_leaves dimension

        return samples

    def extra_repr(self):
        return f"num_features={self.num_features}, num_features_out={self.num_features_out}"
