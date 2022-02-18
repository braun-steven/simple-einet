from typing import List
import torch
import numpy as np

from simple_einet.utils import SamplingContext
from simple_einet.layers import AbstractLayer
from simple_einet.distributions import AbstractLeaf


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
            self, num_features: int, num_features_out: int, num_repetitions, base_leaf: AbstractLeaf, use_em=False
    ):
        """
        Args:
            in_features (int): Number of input features/RVs.
            out_features (int): Number of output features/RVs.
            num_repetitions (int): Number of repetitions.
            base_leaf (Leaf): Base leaf distribution object.
        """

        super().__init__(num_features, num_repetitions=num_repetitions, use_em=use_em)

        self.base_leaf = base_leaf
        self.base_leaf.initialize()
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
        # NOTE: for now, permute channels and feature dim b/c exp family expects [batch, features, channels]
        x = x.permute(0, 2, 1)
        x = self.base_leaf(x, marginalized_scopes)

        # NOTE: This is already done in the exponential fam array?
        # # Factorize input channels
        # x = x.sum(dim=1)

        # Merge scopes by naive factorization
        x = torch.einsum("bicr,ior->bocr", x, self.scopes)

        # assert x.shape == (x.shape[0], self.num_features_out, self.base_leaf.num_leaves, self.num_repetitions)
        assert x.shape == (
            x.shape[0],
            self.num_features_out,
            self.base_leaf.array_shape[0],
            self.num_repetitions,
        )
        return x

    def sample(self, num_samples: int = None, context: SamplingContext = None) -> torch.Tensor:
        # Save original indices_out and set context indices_out to none, such that the out_channel
        # are not filtered in the base_leaf sampling procedure
        indices_out = context.indices_out
        context.indices_out = None
        # samples = self.base_leaf.sample(context=context)
        samples = self.base_leaf.sample(num_samples=context.num_samples)
        # NOTE: exp family array has channel at dim=2 -> permute to dim=1
        samples = samples.permute(0, 2, 1, 3, 4)

        # NOTE: exp family doesn't index repetitions yet, (TODO)

        # Index samples to get the correct repetitions
        r_idxs = context.indices_repetition.view(-1, 1, 1, 1, 1)
        r_idxs = r_idxs.expand(-1, self.base_leaf.num_channels, self.num_features, self.base_leaf.array_shape[0], -1)
        samples = samples.gather(dim=-1, index=r_idxs)
        samples = samples.squeeze(-1)

        num_samples = samples.shape[0]

        # Check that shapes match as expected
        if samples.dim() == 4:
            assert samples.shape == (
                context.num_samples,
                self.base_leaf.num_channels,
                self.num_features,
                self.base_leaf.array_shape[0],
            )
        # elif samples.dim() == 5:
        #     assert self.num_features == samples.shape[1]
        #     assert hasattr(self.base_leaf, "cardinality")
        #     assert samples.shape == (
        #         context.num_samples,
        #         self.base_leaf.out_features,
        #         self.base_leaf.cardinality,
        #         self.base_leaf.num_leaves,
        #     )

        scopes = self.scopes[..., context.indices_repetition].permute(2, 0, 1)
        rnge_in = torch.arange(self.num_features_out, device=samples.device)
        scopes = (scopes * rnge_in).sum(-1).long()
        indices_in_gather = indices_out.gather(dim=1, index=scopes)
        indices_in_gather = indices_in_gather.view(num_samples, 1, -1, 1).expand(
            -1, samples.shape[1], -1, -1
        )
        samples = samples.gather(dim=-1, index=indices_in_gather)
        samples.squeeze_(-1)  # Remove num_leaves dimension

        return samples

    def extra_repr(self):
        return f"num_features={self.num_features}, num_features_out={self.num_features_out}"
