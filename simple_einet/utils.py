from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Tuple

from torch.nn import functional as F
import numpy as np
import torch
from scipy.stats import rankdata
from torch import Tensor, nn
from tqdm.std import tqdm

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

        if evidence is not None:
            # Enter
            for module in spn.modules():
                if hasattr(module, "_enable_input_cache"):
                    module._enable_input_cache()

            _ = spn(evidence, marginalized_scopes)

        # Run in context (nothing needs to be yielded)
        yield

        # Exit
        if evidence is not None:
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

    # Evidence
    evidence: torch.Tensor = None

    # Differentiable
    is_differentiable: bool = False

    # Flag for hard or soft differentiable sampling
    hard: bool = False

    # Temperature for differentiable sampling
    tau: float = 1.0

    # Do MPE at leaves
    mpe_at_leaves: bool = False

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


def get_context(differentiable):
    """
    Get a noop context if differentiable, else torch.no_grad.

    Args:
      differentiable: If the context should allow gradients or not.

    Returns:
      nullcontext if differentialbe=False, else torch.no_grad

    """
    if differentiable:
        return nullcontext()
    else:
        return torch.no_grad()


def diff_sample_one_hot(logits: torch.Tensor, dim: int, mode: str, hard: bool, tau: float) -> torch.Tensor:
    """
    Perform differentiable sampling/mpe on the given input along a specific dimension.

    Modes:
    - "sample": Perform sampling
    - "argmax": Perform mpe

    Args:
      logits(torch.Tensor): Logits from which the sampling should be done.
      dim(int): Dimension along which to sample from.
      mode(str): Mode as described above.
      hard(bool): Whether to perform hard or soft sampling.
      tau(float): Temperature for soft sampling.

    Returns:
      torch.Tensor: Indices encoded as one-hot tensor along the given dimension `dim`.

    """
    if mode == "sample":
        return F.gumbel_softmax(logits=logits, hard=hard, tau=tau, dim=dim)
    elif mode == "argmax":
        # Differentiable argmax (see gumbel softmax trick code)
        y_soft = logits.softmax(dim)
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
        return ret
    else:
        raise Exception(f"Invalid mode option (got {mode}). Must be either 'sample' or 'argmax'.")


def index_one_hot(tensor: torch.Tensor, index: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Index into a given tensor unsing a one-hot encoded index tensor at a specific dimension.

    Example:

    Given array "x = [3 7 5]" and index "2", "x[2]" should return "5".

    Here, index "2" should be one-hot encoded: "2 = [0 0 1]" which we use to
    elementwise multiply the original tensor and then sum up the masked result.

    sum([3 7 5] * [0 1 0]) == sum([0 0 5]) == 5

    The implementation is equivalent to

        torch.sum(tensor * index, dim)

    but uses the einsum operation to reduce the number of operations from two to one.

    Args:
      tensor(torch.Tensor): Tensor which shall be indexed.
      index(torch.Tensor): Indexing tensor.
      dim(int): Dimension at which the tensor should be used index.

    Returns:
      torch.Tensor: Indexed tensor.

    """
    assert (
        tensor.shape[dim] == index.shape[dim]
    ), f"Tensor and index at indexing dimension must be the same size but was tensor.shape[{dim}]={tensor.shape[dim]} and index.shape[{dim}]={index.shape[dim]}"

    assert (
        tensor.dim() == index.dim()
    ), f"Tensor and index number of dimensions must be the same but was tensor.dim()={tensor.dim()} and index.dim()={index.dim()}"

    if __HAS_EINSUM_BROADCASTING and False:
        num_dims = tensor.dim()
        dims = "abcdefghijklmnopqrstuvwxyz"[:num_dims]
        dims_without = dims[:dim] + dims[dim + 1 :]
        einsum_str = f"{dims},{dims}->{dims_without}"
        # print(f"tensor.shape: {tensor.shape}")
        # print(f"index.shape:  {index.shape}")
        # print(f"einsum_str:  {einsum_str}")
        # print(f"dim={dim}")
        return torch.einsum(einsum_str, tensor, index)
    else:
        return torch.sum(tensor * index, dim=dim)
