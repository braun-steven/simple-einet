from typing import Any, Dict, Iterable, List, Tuple

import torch
from torch import distributions as dist
from torch import nn

from simple_einet.layers.distributions.abstract_leaf import AbstractLeaf
from simple_einet.sampling_utils import SamplingContext
from simple_einet.utils import invert_permutation


class MultiDistributionLayer(AbstractLeaf):
    def __init__(
        self,
        scopes_to_dist: List[Tuple[Iterable[int], AbstractLeaf, Dict[str, Any]]],
        num_features: int,
        num_channels: int,
        num_leaves: int,
        num_repetitions: int = 1,
    ):
        """Construct a leaf layer that represents multiple distributions.

        Args:
            scopes_to_dist (List[Tuple[Iterable[int], AbstractLeaf]]): List of lists of indices that represent scopes to a
                specific distribution (AbstractLeaf object).
            num_features (int): Number of input features.
            num_channels (int): Number of input channels.
            num_leaves (int): Number of leaves per distribution per input random variable.
            num_repetitions (int, optional): Number of repetitions. Defaults to 1.
        """
        super().__init__(
            num_features,
            num_channels,
            num_leaves,
            num_repetitions=num_repetitions,
        )

        # Check that the index list covers all features
        all_scopes = []
        for scopes, _, _ in scopes_to_dist:
            for scope in scopes:
                all_scopes.append(scope)

        assert len(all_scopes) == num_features
        scope_list = []
        dists = []
        for scopes, dist_class, dist_kwargs in scopes_to_dist:
            # Construct distribution object
            dist = dist_class(
                num_features=len(scopes),
                num_channels=num_channels,
                num_leaves=num_leaves,
                num_repetitions=num_repetitions,
                **dist_kwargs,
            )
            scope_list.append(scopes)
            dists.append(dist)

        self.scopes = scope_list
        self.dists = nn.ModuleList(dists)

        self.scopes_to_dist = scopes_to_dist
        self.inverted_index = invert_permutation(torch.tensor(all_scopes))

        # Instantiate distributions

        # Check if scopes are already sorted, if so, inversion is not necessary in the forward pass
        self.needs_inversion = all_scopes != list(sorted(all_scopes))

    def forward(self, x, marginalized_scopes: List[int] = None):
        # Collect lls from all distributions
        lls_all = []

        # Forward through all base distributions
        for scope, dist in zip(self.scopes, self.dists):
            x_d = x[:, :, scope]
            lls = dist(x_d, marginalized_scopes=None)
            lls_all.append(lls)

        # Stack along feature dimension
        lls = torch.cat(lls_all, dim=2)

        # If inversion is necessary, permute features to obtain the original order
        if self.needs_inversion:
            lls = lls[:, :, self.inverted_index]

        # Marginalize
        lls = self._marginalize_input(lls, marginalized_scopes)

        return lls

    def sample(self, ctx: SamplingContext) -> torch.Tensor:
        all_samples = []
        indices_out = ctx.indices_out
        for scope, dist in zip(self.scopes, self.dists):
            if ctx.indices_out is not None:
                ctx.indices_out = indices_out[:, scope]
            samples = dist.sample(ctx)
            all_samples.append(samples)

        if ctx.return_leaf_params:
            # Same code as in get_params() -- TODO: Refactor to reuse code
            params = all_samples
            max_num_params = max([p.shape[-1] for p in params])
            for i, p in enumerate(params):
                if p.shape[-1] < max_num_params:
                    # Pad with zeros
                    new_shape = list(p.shape)
                    new_shape[-1] = max_num_params - p.shape[-1]
                    params[i] = torch.cat([p, torch.zeros(new_shape, device=p.device, dtype=p.dtype)], dim=-1)
            samples = torch.cat(params, dim=2)
        else:
            samples = torch.cat(all_samples, dim=2)

        # If inversion is necessary, permute features to obtain the original order
        if self.needs_inversion:
            samples = samples[:, :, self.inverted_index]

        return samples

    def _get_base_distribution(self) -> dist.Distribution:
        raise NotImplementedError("MultiDistributionLayer does not implement _get_base_distribution.")

    def get_params(self):
        """
        Collect params from all distributions and concatenate them along the feature dimension.

        Note: If the number of parameters of the distributions is not equal, the distributions with fewer parameters
        are padded with zeros. That is, get_params().shape[-1] should contain the different paramters of the
        distribution (mu, sigma) for a Normal. In the case of a MultiDistribution of a Bernoulli (a single paramter: p),
        and a Normal (two parameters: mu, sigma) this will lead to the Bernoulli parameters being padded to (p, 0)
        in the last dimension.
        """
        params = [d.get_params() for d in self.dists]
        max_num_params = max([p.shape[-1] for p in params])
        for i, p in enumerate(params):
            if p.shape[-1] < max_num_params:
                # Pad with zeros
                new_shape = list(p.shape)
                new_shape[-1] = max_num_params - p.shape[-1]
                params[i] = torch.cat([p, torch.zeros(new_shape, device=p.device, dtype=p.dtype)], dim=-1)
        return torch.cat(params, dim=2)
