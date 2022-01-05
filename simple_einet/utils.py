from contextlib import contextmanager, nullcontext
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@contextmanager
def provide_evidence(
    spn: nn.Module,
    evidence: torch.Tensor = None,
    marginalized_scopes: torch.Tensor = None,
    requires_grad=False,
):
    """
    Context manager for sampling with evidence. In this context, the SPN graph is reweighted with the likelihoods
    computed using the given evidence.

    Args:
        spn: SPN that is being used to perform the sampling.
        evidence: Provided evidence. The SPN will perform a forward pass prior to entering this contex.
        requires_grad: If False, runs in torch.no_grad() context. (default: False)
    """
    # If no gradients are required, run in no_grad context
    if not requires_grad:
        context = torch.no_grad
    else:
        # Else provide null context
        context = nullcontext

    # Run forward pass in given context
    with context():
        # Enter
        for module in spn.modules():
            if hasattr(module, "_enable_input_cache"):
                module._enable_input_cache()

        if evidence is not None:
            _ = spn(evidence, marginalized_scopes)

        # Run in context (nothing needs to be yielded)
        yield

        # Exit
        for module in spn.modules():
            if hasattr(module, "_enable_input_cache"):
                module._disable_input_cache()


@dataclass
class SamplingContext:
    # Number of samples
    num_samples: int = None

    # Indices into the out_channels dimension
    indices_out: torch.Tensor = None

    # Indices into the repetition dimension
    indices_repetition: torch.Tensor = None

    # MPE flag, if true, will perform most probable explanation sampling
    is_mpe: bool = False

    # Temperature for sampling at the leaves
    temperature_leaves: float = 1.0

    # Temperature for sampling at the einsumlayers
    temperature_sums: float = 1.0

    # Number of repetitions
    num_repetitions: int = None

    def __setattr__(self, key, value):
        if hasattr(self, key):
            super().__setattr__(key, value)
        else:
            raise AttributeError(f"SamplingContext object has no attribute {key}")

    @property
    def is_root(self):
        return self.indices_out == None and self.indices_repetition == None


def invert_permutation(p: torch.Tensor):
    """
    The argument p is assumed to be some permutation of 0, 1, ..., len(p)-1.
    Returns an array s, where s[i] gives the index of i in p.
    Taken from: https://stackoverflow.com/a/25535723, adapted to PyTorch.
    """
    s = torch.empty(p.shape[0], dtype=p.dtype, device=p.device)
    s[p] = torch.arange(p.shape[0], device=p.device)
    return s


def calc_bpd(log_p: Tensor, image_shape: Tuple[int, int, int], has_gauss_dist: bool, n_bins: int) -> float:
    n_pixels = np.prod(image_shape)

    if has_gauss_dist:
        # https://github.com/openai/glow/blob/eaff2177693a5d84a1cf8ae19e8e0441715b82f8/model.py#L172
        bpd = log_p - np.log(n_bins) * n_pixels
        bpd = (-bpd / (np.log(2) * n_pixels)).mean()

    else:
        bpd = log_p - np.log(n_bins) * n_pixels
        bpd = (-bpd / (np.log(2) * n_pixels)).mean()

    return bpd


def dequantize_image(image: Tensor, n_bins: int) -> Tensor:
    return image + torch.rand_like(image) / n_bins


def reduce_bits(image: Tensor, n_bits: int) -> Tensor:
    image = image * 255
    if n_bits < 8:
        image = torch.floor(image / 2 ** (8 - n_bits))

    return image


def preprocess(
    image: Tensor,
    n_bits: int,
    n_bins: int,
    dequantize=True,
    has_gauss_dist=True,
) -> Tensor:
    image = reduce_bits(image, n_bits)
    if has_gauss_dist:
        image = image / n_bins - 0.5
        if dequantize:
            image = dequantize_image(image, n_bins)
    else:
        image = image.long()
    return image

def linearize_index(indices: List[Tuple[int, int]], shape: Tuple[int, int]):
    """

    Args:
      indices: List[Tuple[int,int]]: Index as 2d-coordinates.
      shape: Tuple[int,int]: Shape of the 2d array.

    Returns:
      Linearized index.

    """
    pass

def delinearize_index():
    pass
