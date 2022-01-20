from torch import distributions as dist
import torch
from torch import nn
from simple_einet.type_checks import check_valid

from .abstract_leaf import AbstractLeaf


class Binomial(AbstractLeaf):
    def __init__(
        self,
        num_features: int,
        num_channels: int,
        num_leaves: int,
        num_repetitions: int,
        total_count: int,
    ):
        super().__init__(
            num_features=num_features,
            num_channels=num_channels,
            num_leaves=num_leaves,
            num_repetitions=num_repetitions,
        )

        self.total_count = check_valid(total_count, int, lower_bound=1)

        # Create binomial parameters
        self.probs = nn.Parameter(
            0.5 + torch.rand(1, num_channels, num_features, num_leaves, num_repetitions) * 0.1
        )

    def _get_base_distribution(self):
        # Use sigmoid to ensure, that probs are in valid range
        return dist.Binomial(self.total_count, probs=torch.sigmoid(self.probs))
