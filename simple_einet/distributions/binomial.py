from torch import distributions as dist, nn
import numpy as np
import torchvision.models as models
from simple_einet.utils import SamplingContext
from typing import List, Tuple
import torch
from torch import nn
from simple_einet.type_checks import check_valid

from simple_einet.distributions.abstract_leaf import AbstractLeaf, dist_forward, dist_mode, dist_sample


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


class ConditionalBinomial(AbstractLeaf):
    def __init__(
        self,
        num_features: int,
        num_channels: int,
        num_leaves: int,
        num_repetitions: int,
        total_count: int,
        cond_fn: nn.Module,
    ):
        super().__init__(
            num_features=num_features,
            num_channels=num_channels,
            num_leaves=num_leaves,
            num_repetitions=num_repetitions,
        )

        self.total_count = check_valid(total_count, int, lower_bound=1)
        self.cond_fn = cond_fn

        # Create binomial parameters
        # self.nn = NN(num_leaves, num_repetitions=num_repetitions)
        # self.nn = Network(
        #     shape=(1, num_channels, num_features // 2, num_leaves, num_repetitions),
        #     num_layers=4,
        #     activation_fn=nn.Tanh(),
        # )

        self.probs_unconditioned = nn.Parameter(
            0.5 + torch.rand(1, num_channels, num_features // 2, num_leaves, num_repetitions) * 0.1
        )

    def get_cond_dist(self, x_cond):
        hw = int(np.sqrt(x_cond.shape[2]))
        x_cond_shape = x_cond.shape
        probs = self.cond_fn(x_cond.view(-1, x_cond.shape[1], hw, hw))
        probs = probs.view(x_cond_shape[0], x_cond_shape[1], self.num_leaves, self.num_repetitions,
                           hw * hw)
        probs = probs.permute(0, 1, 4, 2, 3)
        probs_unc = self.probs_unconditioned.expand(x_cond.shape[0], -1, -1, -1, -1)
        probs = torch.cat((probs, probs_unc), dim=2)
        d = dist.Binomial(self.total_count, logits=probs)
        return d

    def forward(self, x, marginalized_scopes: List[int]):
        x_cond = x[:, :, x.shape[2] // 2 :, None, None]
        d = self.get_cond_dist(x_cond)
        x = dist_forward(d, x)

        x = self._marginalize_input(x, marginalized_scopes)

        return x

    def sample(self, num_samples: int = None, context: SamplingContext = None) -> torch.Tensor:
        ev = context.evidence
        x_cond = ev[:, :, ev.shape[2] // 2 :, None, None]
        d = self.get_cond_dist(x_cond)

        # Sample from the specified distribution
        if context.is_mpe:
            samples = dist_mode(d, context)
        else:
            samples = d.sample()

        num_samples, num_channels, num_features, num_leaves, num_repetitions = samples.shape

        # Index samples to get the correct repetitions
        r_idxs = context.indices_repetition.view(-1, 1, 1, 1, 1)
        r_idxs = r_idxs.expand(-1, num_channels, num_features, num_leaves, -1)
        samples = samples.gather(dim=-1, index=r_idxs)
        samples = samples.squeeze(-1)

        # If parent index into out_channels are given
        if context.indices_out is not None:
            # Choose only specific samples for each feature/scope
            samples = torch.gather(samples, dim=2, index=context.indices_out.unsqueeze(-1)).squeeze(
                -1
            )

        return samples

    def _get_base_distribution(self) -> dist.Distribution:
        raise NotImplementedError("This should not happen.")


class Linear(nn.Module):
    def __init__(self, shape: List[int]) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.rand(shape))
        self.bias = nn.Parameter(torch.rand(shape))

    def forward(self, x):
        return self.weights * x + self.bias


class Network(nn.Module):
    def __init__(self, shape: List[int], num_layers: int, activation_fn) -> None:
        super().__init__()

        self.resnet = models.resnet18()
        # layers = []
        # for i in range(num_layers):
        #     layers.append(Linear(shape))

        #     # Skip activation after last layer
        #     if i < num_layers - 1:
        #         layers.append(activation_fn)

        # self.seq = nn.Sequential(*layers)

    def forward(self, x):
        breakpoint()
        return self.resnet(x)
