import torch
from torch.nn import functional as F

from simple_einet.sampling_utils import SamplingContext


def get_sampling_context(layer, num_samples: int, is_differentiable: bool = False):
    if is_differentiable:
        indices_out = torch.randint(low=0, high=layer.num_sums_out, size=(num_samples, layer.num_features_out))
        one_hot_indices_out = F.one_hot(indices_out, num_classes=layer.num_sums_out).float()
        indices_repetition = torch.randint(low=0, high=layer.num_repetitions, size=(num_samples,))
        one_hot_indices_repetition = F.one_hot(indices_repetition, num_classes=layer.num_repetitions).float()
        one_hot_indices_out.requires_grad_(True)
        one_hot_indices_repetition.requires_grad_(True)
        return SamplingContext(
            num_samples=num_samples,
            indices_out=one_hot_indices_out,
            indices_repetition=one_hot_indices_repetition,
            is_differentiable=True,
        )
    else:
        return SamplingContext(
            num_samples=num_samples,
            indices_out=torch.randint(low=0, high=layer.num_sums_out, size=(num_samples, layer.num_features_out)),
            indices_repetition=torch.randint(low=0, high=layer.num_repetitions, size=(num_samples,)),
            is_differentiable=False,
        )
