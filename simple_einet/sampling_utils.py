from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from enum import Enum

import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

import simple_einet

from simple_einet.utils import __HAS_EINSUM_BROADCASTING


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
    """Dataclass for representing the context in which sampling operations occur."""

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


class DiffSampleMethod(str, Enum):
    """Enum for differentiable sampling methods."""

    SIMPLE = "SIMPLE"
    GUMBEL = "GUMBEL"


def sample_gumbel(shape, eps=1e-20, device="cpu"):
    """
    Samples from a Gumbel distribution with the given shape.

    Args:
        shape (tuple): The shape of the desired output tensor.
        eps (float, optional): A small value to avoid numerical instability. Defaults to 1e-20.

    Returns:
        torch.Tensor: A tensor of the specified shape sampled from a Gumbel distribution.
    """
    U = torch.rand(shape, device=device)
    g = -torch.log(-torch.log(U + eps) + eps)
    return g


def SIMPLE(logits, dim: int) -> torch.Tensor:
    """
    Sample from the distribution using SIMPLE[1].

    [1] https://github.com/UCLA-StarAI/SIMPLE, https://arxiv.org/abs/2210.01941

    Args:
        logits (torch.Tensor): Input tensor of shape [*, n_class].
        dim (int): Dimension along which to sample.

    Returns:
        torch.Tensor: Output tensor of shape [*, n_class], an one-hot vector.
    """
    y = F.softmax(logits, dim=dim)
    y = logits + sample_gumbel(logits.size(), device=logits.device)
    y_perturbed = F.softmax(y, dim=dim)

    shape = y.size()
    index = y_perturbed.max(dim=dim, keepdim=True)[1]
    y_hard = torch.zeros_like(y_perturbed)
    y_hard.scatter_(dim, index, 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y


def sample_categorical_differentiably(
    logits: torch.Tensor,
    dim: int,
    is_mpe: bool,
    hard: bool,
    tau: float,
    method=DiffSampleMethod.SIMPLE,
) -> torch.Tensor:
    """
    Perform differentiable sampling/mpe on the given input along a specific dimension.

    Args:
      logits(torch.Tensor): Logits from which the sampling should be done.
      dim(int): Dimension along which to sample from.
      is_mpe(bool): Whether to perform MPE sampling.
      hard(bool): Whether to perform hard or soft sampling.
      tau(float): Temperature for soft sampling.
      method(DiffSampleMethod): Method to use for differentiable sampling. Must be either DiffSampleMethod.SIMPLE or DiffSampleMethod.GUMBEL.

    Returns:
      torch.Tensor: Indices encoded as one-hot tensor along the given dimension `dim`.
    """
    if not is_mpe:
        if method == DiffSampleMethod.SIMPLE:
            return SIMPLE(logits, dim=dim)
        elif method == DiffSampleMethod.GUMBEL:
            return F.gumbel_softmax(logits=logits, hard=hard, tau=tau, dim=dim)
        else:
            raise Exception(f"Invalid method option (got {method}). Must be either 'SIMPLE' or 'GUMBEL'.")
    else:
        if method == DiffSampleMethod.SIMPLE:
            return SIMPLE(logits, dim=dim)
        elif method == DiffSampleMethod.GUMBEL:
            # Differentiable argmax (see gumbel softmax trick code in pytorch)
            y_soft = logits.softmax(dim)
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
            return ret
        else:
            raise Exception(f"Invalid method option (got {method}). Must be either 'SIMPLE' or 'GUMBEL'.")


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


def init_einet_stats(einet: "Einet", dataloader: torch.utils.data.DataLoader):
    """
    Initializes the statistics of the Einet model using the given dataloader.

    Args:
    - einet (Einet): The Einet model to initialize.
    - dataloader (torch.utils.data.DataLoader): The dataloader to use for initialization.

    Returns: None
    """
    stats_mean = None
    stats_std = None

    # Compute mean and std
    for batch in tqdm(dataloader, desc="Leaf Parameter Initialization"):
        data, label = batch
        if stats_mean == None:
            stats_mean = data.mean(dim=0)
            stats_std = data.std(dim=0)
        else:
            stats_mean += data.mean(dim=0)
            stats_std += data.std(dim=0)

    # Normalize
    stats_mean /= len(dataloader)
    stats_std /= len(dataloader)

    from simple_einet.distributions.normal import Normal
    from simple_einet.einet import Einet, EinetMixture

    # Set leaf parameters for normal distribution
    if einet.config.leaf_type == Normal:
        if type(einet) == Einet:
            einets = [einet]
        elif type(einet) == EinetMixture:
            einets = einet.einets
        else:
            raise ValueError(f"Invalid einet type: {type(einet)} -- must be Einet or EinetMixture.")

        # Reshape to match leaf parameters
        stats_mean_v = (
            stats_mean.view(-1, 1, 1)
            .repeat(1, einets[0].config.num_leaves, einets[0].config.num_repetitions)
            .view_as(einets[0].leaf.base_leaf.means)
        )
        stats_std_v = (
            stats_std.view(-1, 1, 1)
            .repeat(1, einets[0].config.num_leaves, einets[0].config.num_repetitions)
            .view_as(einets[0].leaf.base_leaf.log_stds)
        )

        # Set leaf parameters
        for net in einets:
            # Add noise to ensure that values are not completely equal along repetitions and einets
            net.leaf.base_leaf.means.data = stats_mean_v + 0.1 * torch.normal(
                torch.zeros_like(stats_mean_v), torch.std(stats_mean_v)
            )
            net.leaf.base_leaf.log_stds.data = torch.log(
                stats_std_v
                + 1e-3
                + torch.clamp(0.1 * torch.normal(torch.zeros_like(stats_std_v), torch.std(stats_std_v)), min=0.0)
            )
