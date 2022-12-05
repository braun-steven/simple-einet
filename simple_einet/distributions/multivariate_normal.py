from typing import List

import numpy as np
import torch
from simple_einet.distributions.abstract_leaf import AbstractLeaf, dist_mode
from simple_einet.type_checks import check_valid
from simple_einet.utils import SamplingContext
from torch import distributions as dist
from torch import nn
from torch.nn import functional as F


class MultivariateNormal(AbstractLeaf):
    """Multivariate Gaussian layer."""

    def __init__(
        self,
        in_features: int,
        out_channels: int,
        cardinality: int,
        num_repetitions: int = 1,
        dropout=0.0,
        min_sigma: float = 0.1,
        max_sigma: float = 1.0,
    ):
        """Creat a multivariate gaussian layer.

        Args:
            out_channels: Number of parallel representations for each input feature.
            in_features: Number of input features.
            num_repetitions: Number of parallel repetitions of this layer.
            cardinality: Number of features covered.

        """
        # TODO: Fix for num_repetitions
        super().__init__(in_features, out_channels, num_repetitions, dropout, cardinality)
        raise NotImplementedError("MultivariateNormal has not been adapted to the new implementation yet - sorry.")
        self._pad_value = in_features % cardinality
        self.out_features = np.ceil(in_features / cardinality).astype(int)
        self._n_dists = np.ceil(in_features / cardinality).astype(int)

        self.min_sigma = check_valid(min_sigma, float, 0.0, max_sigma)
        self.max_sigma = check_valid(max_sigma, float, min_sigma)

        # Create gaussian means and covs
        self.means = nn.Parameter(torch.randn(out_channels * self._n_dists * self.num_repetitions, cardinality))

        # Generate covariance matrix via the cholesky decomposition: s = A'A where A is a triangular matrix
        # Further ensure, that diag(a) > 0 everywhere, such that A has full rank
        rand = torch.zeros(out_channels * self._n_dists * self.num_repetitions, cardinality, cardinality)

        for i in range(cardinality):
            rand[:, i, i] = 1.0

        rand = rand + torch.randn_like(rand) * 1e-1

        # Make matrices triangular and remove diagonal entries
        cov_tril_wo_diag = rand.tril(diagonal=-1)
        cov_tril_wi_diag = torch.rand(out_channels * self._n_dists * self.num_repetitions, cardinality, cardinality)

        self.cov_tril_wo_diag = nn.Parameter(cov_tril_wo_diag)
        self.cov_tril_wi_diag = nn.Parameter(cov_tril_wi_diag)
        # self._mv = dist.MultivariateNormal(loc=self.means, scale_tril=self.triangular)
        # Reassign means since mv __init__ creates a copy and thus would loose track for autograd
        # self._mv.loc.requires_grad_(True)
        # self.means = nn.Parameter(self._mv.loc)

        self.out_shape = f"(N, {self.out_features}, {self.num_leaves})"

    def forward(self, x: torch.Tensor, marginalized_scopes: List[int]) -> torch.Tensor:
        batch_size = x.shape[0]

        # Pad dummy variable via reflection
        if self._pad_value != 0:
            x = F.pad(x, pad=[0, 0, 0, self._pad_value], mode="reflect")

        # Make room for repetitions: [n, 1, d]
        x = x.unsqueeze(1)

        # Split features into groups
        x = x.view(batch_size, 1, 1, self._n_dists, self.cardinality)  # [n, 1, 1, d/k, k]

        # Repeat groups by number of output_channels and number of repetitions
        x = x.repeat(1, self.num_repetitions, self.num_leaves, 1, 1)  #  [n, r, oc, d/k, k]

        # Merge groups and repetitions
        x = x.view(
            batch_size, self.num_repetitions * self.num_leaves * self._n_dists, self.cardinality
        )  #  [n, r * d/k * oc, k]

        # Compute multivariate gaussians
        # Output shape: [n, out_channels, d / cardinality]
        mv = self._get_base_distribution()
        x = mv.log_prob(x)  #  [n, r * d/k * oc]
        x = x.view(batch_size, self.num_repetitions, self.num_leaves, self._n_dists)  # [n, r, oc, d/k]
        x = x.permute(0, 3, 2, 1)  # [n, d/k, oc, r]

        # Marginalize and apply dropout
        x = self._marginalize_input(x, marginalized_scopes)
        x = self._apply_dropout(x)

        return x

    def sample(self, num_samples: int = None, context: SamplingContext = None) -> torch.Tensor:
        mv = self._get_base_distribution()

        # Sample from the specified distribution
        if context.is_mpe:
            samples = dist_mode(mv, context)
        else:
            samples = mv.sample(sample_shape=(context.num_samples,))

            samples = samples.view(
                context.num_samples,
                self.num_repetitions,
                self.num_leaves,
                self._n_dists,
                self.cardinality,
            )

        num_samples, num_repetitions, out_channels, n_dists, cardinality = samples.shape

        # Filter each sample by its specific repetition
        tmp = torch.zeros(
            num_samples,
            out_channels,
            n_dists,
            cardinality,
            device=context.indices_repetition.device,
        )
        for i in range(num_samples):
            tmp[i] = samples[i, context.indices_repetition[i], ...]

        samples = tmp  # [n, oc, d/k, k]

        samples = samples.view(
            context.num_samples,
            self.num_leaves,
            self._n_dists,
            self.cardinality,
        )
        samples = samples.permute(0, 2, 3, 1)

        return samples

    def _get_base_distribution(self):

        if self.min_sigma < self.max_sigma:
            # scale diag to [min_sigma, max_sigma]
            cov_diag = self.cov_tril_wi_diag
            sigma_ratio = torch.sigmoid(cov_diag)
            cov_diag = self.min_sigma + (self.max_sigma - self.min_sigma) * sigma_ratio
            cov_diag = cov_diag.tril()

            # scale tril to [-max_sigma, max_sigma]
            cov_tril = self.cov_tril_wo_diag
            sigma_ratio = torch.sigmoid(cov_tril)
            cov_tril = -1 * self.max_sigma + 2 * self.max_sigma * sigma_ratio
            cov_tril = cov_tril.tril(-1)

        else:
            cov_tril = self.cov_tril_wo_diag.tril(-1)
            cov_diag = self.cov_tril_wi_diag.tril().sigmoid()

        scale_tril = cov_tril + cov_diag
        mv = dist.MultivariateNormal(loc=self.means, scale_tril=scale_tril)

        # ic(cov_diag.mean(0))
        # ic(cov_tril.mean(0))
        return mv
