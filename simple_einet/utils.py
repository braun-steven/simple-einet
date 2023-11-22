from typing import Tuple

import numpy as np
import torch
from torch import Tensor

# Assert that torch.einsum broadcasting is available check for torch version >= 1.8.0
try:
    __TORCHVERSION = [int(v) for v in torch.__version__.split(".")]
    __V_MAJOR = __TORCHVERSION[0]
    __V_MINOR = __TORCHVERSION[1]
    if __V_MAJOR == 0:
        __HAS_EINSUM_BROADCASTING = False
    elif __V_MAJOR == 1 and __V_MINOR < 8:
        __HAS_EINSUM_BROADCASTING = False
    else:
        __HAS_EINSUM_BROADCASTING = True
except:
    __HAS_EINSUM_BROADCASTING = False


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
    """
    Calculates the bits per dimension (BPD) for a given log probability tensor.

    Args:
        log_p (Tensor): The log probability tensor.
        image_shape (Tuple[int, int, int]): The shape of the image.
        has_gauss_dist (bool): Whether the distribution is Gaussian or not.
        n_bins (int): The number of bins.

    Returns:
        float: The bits per dimension (BPD) value.
    """
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
