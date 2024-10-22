from typing import Tuple

import numpy as np
import torch
from scipy.stats import rankdata
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
    return torch.argsort(p)


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


def rdc(x, y, f=np.sin, k=20, s=1 / 6.0, n=1):
    """

    Source: https://github.com/garydoranjr/rdc/blob/master/rdc/rdc.py

    Computes the Randomized Dependence Coefficient
    x,y: numpy arrays 1-D or 2-D
         If 1-D, size (samples,)
         If 2-D, size (samples, variables)
    f:   function to use for random projection
    k:   number of random projections to use
    s:   scale parameter
    n:   number of times to compute the RDC and
         return the median (for stability)
    According to the paper, the coefficient should be relatively insensitive to
    the settings of the f, k, and s parameters.
    """
    if n > 1:
        values = []
        for i in range(n):
            try:
                values.append(rdc(x, y, f, k, s, 1))
            except np.linalg.linalg.LinAlgError:
                pass
        return np.median(values)

    if len(x.shape) == 1:
        x = x.reshape((-1, 1))
    if len(y.shape) == 1:
        y = y.reshape((-1, 1))

    # Copula Transformation
    cx = np.column_stack([rankdata(xc, method="ordinal") for xc in x.T]) / float(x.size)
    cy = np.column_stack([rankdata(yc, method="ordinal") for yc in y.T]) / float(y.size)

    # Add a vector of ones so that w.x + b is just a dot product
    O = np.ones(cx.shape[0])
    X = np.column_stack([cx, O])
    Y = np.column_stack([cy, O])

    # Random linear projections
    Rx = (s / X.shape[1]) * np.random.randn(X.shape[1], k)
    Ry = (s / Y.shape[1]) * np.random.randn(Y.shape[1], k)
    X = np.dot(X, Rx)
    Y = np.dot(Y, Ry)

    # Apply non-linear function to random projections
    fX = f(X)
    fY = f(Y)

    # Compute full covariance matrix
    C = np.cov(np.hstack([fX, fY]).T)

    # Due to numerical issues, if k is too large,
    # then rank(fX) < k or rank(fY) < k, so we need
    # to find the largest k such that the eigenvalues
    # (canonical correlations) are real-valued
    k0 = k
    lb = 1
    ub = k
    while True:
        # Compute canonical correlations
        Cxx = C[:k, :k]
        Cyy = C[k0 : k0 + k, k0 : k0 + k]
        Cxy = C[:k, k0 : k0 + k]
        Cyx = C[k0 : k0 + k, :k]

        eigs = np.linalg.eigvals(np.dot(np.dot(np.linalg.pinv(Cxx), Cxy), np.dot(np.linalg.pinv(Cyy), Cyx)))

        # Binary search if k is too large
        if not (np.all(np.isreal(eigs)) and 0 <= np.min(eigs) and np.max(eigs) <= 1):
            ub -= 1
            k = (ub + lb) // 2
            continue
        if lb == ub:
            break
        lb = k
        if ub == lb + 1:
            k = ub
        else:
            k = (ub + lb) // 2

    return np.sqrt(np.max(eigs))
