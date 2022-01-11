from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from scipy.stats import rankdata
from torch import Tensor, nn
from tqdm.std import tqdm



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


def calc_bpd(
    log_p: Tensor, image_shape: Tuple[int, int, int], has_gauss_dist: bool, n_bins: int
) -> float:
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

        eigs = np.linalg.eigvals(
            np.dot(np.dot(np.linalg.pinv(Cxx), Cxy), np.dot(np.linalg.pinv(Cyy), Cyx))
        )

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


if __name__ == '__main__':

    from data import get_dataset
    dataset = get_dataset(dataset_name="mnist", data_dir="/home/tak/data", train=True)
    data = dataset.data.view(dataset.data.shape[0], -1)
    D = data.shape[1]
    for i in range(10):
        # idx_x = torch.randperm(D)[:D // 2]
        # idx_y = [i for i in range(D) if i not in idx_x]

        # digit_x = 0
        # digit_y = 1
        # mask_x = dataset.targets == digit_x
        # mask_y = dataset.targets == digit_y

        # x = data[mask_x][:4000, idx_x].numpy()
        # y = data[mask_y][:4000, idx_y].numpy()
        x = data[:1000].numpy()

        coeffs = np.zeros((D, D))

        for i in tqdm(range(D)):
            for j in range(D):
                coeffs[i, j] = rdc(x[:, i], x[:, j], k=1)


        print(coeffs)
