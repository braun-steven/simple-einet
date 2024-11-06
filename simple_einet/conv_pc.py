#!/usr/bin/env python3


from simple_einet.layers.distributions.abstract_leaf import AbstractLeaf
from simple_einet.type_checks import check_valid
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.nn import functional as F
from torch.profiler import record_function

from simple_einet.abstract_layers import logits_to_log_weights
from simple_einet.data import Shape, get_data_shape
from simple_einet.layers.distributions.binomial import Binomial
from simple_einet.sampling_utils import index_one_hot, sample_categorical_differentiably


def prof(name):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with record_function(name):
                return func(*args, **kwargs)

        return wrapper

    return decorator


@dataclass
class SamplingContext:
    """Dataclass for representing the context in which sampling operations occur."""

    # Number of samples
    num_samples: int = None

    # Indices into the out_channels dimension
    indices_out: torch.Tensor = None

    indices_repetition: torch.Tensor = None

    # MPE flag, if true, will perform most probable explanation sampling
    is_mpe: bool = False

    # Temperature for sampling at the leaves
    temperature_leaves: float = 1.0

    # Temperature for sampling at the einsumlayers
    temperature_sums: float = 1.0

    # Evidence
    evidence: torch.Tensor = None

    # Differentiable
    is_differentiable: bool = False

    # Temperature for differentiable sampling
    tau: float = 1.0

    # Do MPE at leaves
    mpe_at_leaves: bool = False

    # Return leaf distribution instead of samples
    return_leaf_params: bool = False

    def __setattr__(self, key, value):
        if hasattr(self, key):
            super().__setattr__(key, value)
        else:
            raise AttributeError(f"SamplingContext object has no attribute {key}")

    def __repr__(self) -> str:
        d = self.__dict__
        d2 = {}
        d2["evidence"] = d["evidence"].shape if d["evidence"] is not None else None
        d2["indices_out"] = d["indices_out"].shape if d["indices_out"] is not None else None
        return "SamplingContext(" + ", ".join([f"{k}={v}" for k, v in d2.items()]) + ")"


@contextmanager
def sampling_context(
    model: nn.Module,
    evidence: torch.Tensor | None = None,
    requires_grad=False,
    seed=None,
):
    """
    Context manager for sampling.

    If evidence is provdied, the SPN graph is reweighted with the likelihoods computed using the given evidence.

    Args:
        spn: SPN that is being used to perform the sampling.
        evidence: Provided evidence. The SPN will perform a forward pass prior to entering this contex.
        requires_grad: If False, runs in torch.no_grad() context. (default: False)
        marginalized_scopes: Scopes to marginalize. (default: None)
        seed: Seed to use for sampling. (default: None)

    """
    # If no gradients are required, run in no_grad context
    if not requires_grad:
        context = torch.no_grad
    else:
        # Else provide null context
        context = nullcontext

    if seed is not None:
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None

        torch.manual_seed(seed)

    # Run forward pass in given context
    with context():
        if evidence is not None:
            # Enter
            for module in model.modules():
                if hasattr(module, "_enable_input_cache"):
                    module._enable_input_cache()

            _ = model.log_likelihood(x=evidence)

        # Run in context (nothing needs to be yielded)
        yield

        # Exit
        if evidence is not None:
            for module in model.modules():
                if hasattr(module, "_enable_input_cache"):
                    module._disable_input_cache()

    if seed is not None:
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state)


class SumConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.logits = nn.Parameter(torch.log(torch.rand(out_channels, in_channels, kernel_size, kernel_size)))

        # Input cache
        self._is_input_cache_enabled = False
        self._input_cache: dict[str, Tensor] = {}

    def forward(self, x):

        if self._is_input_cache_enabled:
            self._input_cache["x"] = x

        N, C, H, W = x.size()
        K = self.kernel_size

        # x_og = x.clone()

        assert H % K == 0, f"Input height must be divisible by kernel size. Height was {H} and kernel size was {K}"
        assert W % K == 0, f"Input width must be divisible by kernel size. Width was {W} and kernel size was {K}"

        logits = self.logits
        log_weights = F.log_softmax(logits, dim=1)
        log_weights = log_weights.unsqueeze(0)

        # Make two new dimensions, such that the patches of KxK are now stacked
        x = x.view(N, C, H // K, K, W // K, K)
        x = x.permute(0, 1, 2, 4, 3, 5)  # 0 1 2 3 4 5 -> 0 1 2 4 3 5

        # Make space for out_channels
        x = x.view(N, 1, C, H // K, W // K, K, K)
        assert x.size() == (N, 1, C, H // K, W // K, K, K)

        # Make space in log_weights for H // kernel_size and W // kernel_size
        log_weights = log_weights.view(1, self.out_channels, self.in_channels, 1, 1, K, K)

        # Weighted sum over input channels
        x = torch.logsumexp(x + log_weights, dim=2)  # 0 1 2 4 3 5 -> 0 1 3 2 4

        # Invert permutation
        x = x.permute(0, 1, 2, 4, 3, 5)  # 0 1 2 4 3 5 -> 0 1 2 3 4 5
        x = x.contiguous().view(N, self.out_channels, H, W)
        return x
        # # return x
        #
        # ###############################################
        # # ALTERNATIVE IMPLEMENTATION, probably slower #
        # ###############################################
        # # Repeat the sum layer weight patch to match the input size in H/W
        #
        # breakpoint()
        # x = x_og
        # log_weights = self.logits.log_softmax(dim=1)
        # log_weights = log_weights.unsqueeze(0)
        # log_weights = log_weights.repeat(1, 1, 1, H // K, W // K)
        # assert log_weights.size() == (1, self.out_channels, self.in_channels, H, W)
        #
        # # Adjust input shape to match the weight shape (make space for out_channels)
        # x = x.view(N, 1, C, H, W)
        #
        # # Weighted sum over input channels
        # x = torch.logsumexp(x + log_weights, dim=2)
        #
        # assert x.size() == (N, self.out_channels, H, W)
        # return x

    def sample(self, ctx: SamplingContext) -> SamplingContext:
        # Select weights of this layer based on parent sampling path
        log_weights = self._select_weights(ctx, self.logits)

        # If evidence is given, adjust the weights with the likelihoods of the observed paths
        if self._is_input_cache_enabled and len(self._input_cache) > 0:
            log_weights = self._condition_weights_on_evidence(ctx, log_weights)

        # Sample/mpe from the logweights
        indices = self._sample_from_weights(ctx, log_weights)

        ctx.indices_out = indices
        return ctx

    def _condition_weights_on_evidence(self, ctx, log_weights):
        input_cache_x = self._input_cache["x"]
        lls = input_cache_x
        log_prior = log_weights
        log_posterior = log_prior + lls
        log_posterior = log_posterior - torch.logsumexp(log_posterior, dim=1, keepdim=True)
        log_weights = log_posterior
        return log_weights

    def _select_weights(self, ctx, logits):
        N = ctx.num_samples
        Co, Ci, K, K = logits.shape
        H, W = ctx.indices_out.shape[2], ctx.indices_out.shape[3]

        # Index sums_out
        logits = logits.unsqueeze(0)  # make space for batch dim
        p_idxs = ctx.indices_out.unsqueeze(2)
        x = p_idxs

        x = x.view(N, Co, H // K, K, W // K, K)
        x = x.permute(0, 1, 2, 4, 3, 5)  # 0 1 2 3 4 5 -> 0 1 2 4 3 5

        # Make space for in_channels
        x = x.view(N, Co, 1, H // K, W // K, K, K)
        assert x.size() == (N, Co, 1, H // K, W // K, K, K)

        # Make space in log_weights for H // kernel_size and W // kernel_size
        logits = logits.view(1, Co, Ci, 1, 1, K, K)

        # Index into the "num_sums_out" dimension
        logits = index_one_hot(logits, index=x, dim=1)

        assert logits.shape == (N, Ci, H // K, W // K, K, K)

        # Revert permutations etc
        logits = logits.permute(0, 1, 2, 4, 3, 5)  # 0 1 2 4 3 5 -> 0 1 2 3 4 5
        logits = logits.contiguous().view(N, Ci, H, W)

        log_weights = logits_to_log_weights(logits, dim=2, temperature=ctx.temperature_sums)
        return log_weights
        log_weights_2 = log_weights

        logits = logits_og

        # Index sums_out
        logits = logits.unsqueeze(0)  # make space for batch dim
        p_idxs = ctx.indices_out.unsqueeze(2)  # make space for in_channels

        # Repeat the sum layer weight patch to match the input size in H/W
        logits = logits.repeat(1, 1, 1, H // K, W // K)

        # Index into the "out_channels" dimension
        logits = index_one_hot(logits, index=p_idxs, dim=1)

        log_weights = logits_to_log_weights(logits, dim=1, temperature=ctx.temperature_sums)

        return log_weights

    def _sample_from_weights(self, ctx, log_weights):
        indices = sample_categorical_differentiably(
            dim=1, is_mpe=ctx.is_mpe, hard=False, tau=ctx.tau, log_weights=log_weights
        )
        return indices

    def _enable_input_cache(self):
        self._is_input_cache_enabled = True
        self._input_cache = {}

    def _disable_input_cache(self):
        self._is_input_cache_enabled = False
        self._input_cache.clear()


class SumLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, height: int, width: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.height = height
        self.width = width
        self.logits = nn.Parameter(torch.log(torch.rand(out_channels, in_channels, height, width)))

        # Input cache
        self._is_input_cache_enabled = False
        self._input_cache: dict[str, Tensor] = {}

    def forward(self, x):

        if self._is_input_cache_enabled:
            self._input_cache["x"] = x

        logits = self.logits
        log_weights = F.log_softmax(logits, dim=1)
        log_weights = log_weights.unsqueeze(0)

        # Make space for out_channels in x
        x = x.unsqueeze(1)

        # Weighted sum over input channels
        x = torch.logsumexp(x + log_weights, dim=2)

        return x

    def _enable_input_cache(self):
        self._is_input_cache_enabled = True
        self._input_cache = {}

    def _disable_input_cache(self):
        self._is_input_cache_enabled = False
        self._input_cache.clear()

    def sample(self, ctx: SamplingContext) -> SamplingContext:
        # Select weights of this layer based on parent sampling path
        log_weights = self._select_weights(ctx, self.logits)

        # If evidence is given, adjust the weights with the likelihoods of the observed paths
        if self._is_input_cache_enabled and len(self._input_cache) > 0:
            log_weights = self._condition_weights_on_evidence(ctx, log_weights)

        # Sample/mpe from the logweights
        indices = self._sample_from_weights(ctx, log_weights)

        ctx.indices_out = indices
        return ctx

    def _select_weights(self, ctx, logits):
        # Index sums_out
        logits = logits.unsqueeze(0)  # make space for batch dim
        p_idxs = ctx.indices_out.unsqueeze(2)
        x = p_idxs

        # Index into the "num_sums_out" dimension
        logits = index_one_hot(logits, index=x, dim=1)

        log_weights = logits_to_log_weights(logits, dim=1, temperature=ctx.temperature_sums)

        return log_weights

    def _sample_from_weights(self, ctx, log_weights):
        indices = sample_categorical_differentiably(
            dim=1, is_mpe=ctx.is_mpe, hard=False, tau=ctx.tau, log_weights=log_weights
        )
        return indices

    def _condition_weights_on_evidence(self, ctx, log_weights):
        input_cache_x = self._input_cache["x"]
        lls = input_cache_x
        log_prior = log_weights
        log_posterior = log_prior + lls
        log_posterior = log_posterior - torch.logsumexp(log_posterior, dim=1, keepdim=True)
        log_weights = log_posterior
        return log_weights

    def __repr__(self):
        return f"SumLayer(ic={self.in_channels}, oc={self.out_channels}, h={self.height}, w={self.width})"


class ProdConv(nn.Module):
    def __init__(self, kernel_size_h: int, kernel_size_w: int):
        super().__init__()
        self.kernel_size_h = kernel_size_h
        self.kernel_size_w = kernel_size_w

        # Input cache
        self._is_input_cache_enabled = False
        self._input_cache: dict[str, Tensor] = {}

    def forward(self, x):
        # Use a convolution with depthwise separable filters and ones as weights to simulate patch based product nodes
        # This is equivalent to a product node in the circuit
        N, C, H, W = x.size()
        Kh = self.kernel_size_h
        Kw = self.kernel_size_w

        assert H % Kh == 0, f"Input height must be divisible by kernel size. Height was {H} and kernel size was {Kh}"
        assert W % Kw == 0, f"Input width must be divisible by kernel size. Width was {W} and kernel size was {Kw}"

        # Construct a kernel with ones as weights
        ones = torch.ones(C, 1, Kh, Kw, device=x.device)

        # Apply the kernel to the input
        x = F.conv2d(x, ones, groups=C, stride=(Kh, Kw), padding=0, bias=None)

        return x

    def _enable_input_cache(self):
        self._is_input_cache_enabled = True
        self._input_cache = {}

    def _disable_input_cache(self):
        self._is_input_cache_enabled = False
        self._input_cache.clear()

    def sample(self, ctx: SamplingContext):
        # Use repeat interleave to perform the following operation:
        #           1 1 2 2
        # 1 2  -\   1 1 2 2
        # 3 4  -/   3 3 4 4
        #           3 3 4 4
        idxs = ctx.indices_out

        idxs = torch.repeat_interleave(idxs, repeats=self.kernel_size_h, dim=-2)
        idxs = torch.repeat_interleave(idxs, repeats=self.kernel_size_w, dim=-1)
        ctx.indices_out = idxs
        return ctx

    def __repr__(self):
        return f"ProdConv(kh={self.kernel_size_h}, kw={self.kernel_size_w})"


class SumProdLayer(nn.Module):
    """Combines a SumLayer and a ProdConv in one layer. Implements forward and sample."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        height: int,
        width: int,
        kernel_size_h: int,
        kernel_size_w: int,
        order="sum-prod",
    ):
        super().__init__()
        if order == "sum-prod":
            # self.sum_layer = SumConv(in_channels, out_channels, kernel_size=kernel_size_w)
            self.sum_layer = SumLayer(in_channels, out_channels, height, width)
            self.prod_conv = ProdConv(kernel_size_h, kernel_size_w)
            self.num_features_out = height // kernel_size_h * width // kernel_size_w
            self.out_channels = out_channels
        elif order == "prod-sum":
            self.prod_conv = ProdConv(kernel_size_h, kernel_size_w)
            self.sum_layer = SumLayer(in_channels, out_channels, height // kernel_size_h, width // kernel_size_w)

        self.order = order

        self.num_features_out = height // kernel_size_h * width // kernel_size_w
        self.out_channels = out_channels

    def forward(self, x):
        if self.order == "sum-prod":
            x = self.sum_layer(x)
            x = self.prod_conv(x)
        elif self.order == "prod-sum":
            x = self.prod_conv(x)
            x = self.sum_layer(x)
        else:
            raise ValueError(f"Order {self.order} not supported")
        return x

    def sample(self, ctx: SamplingContext):
        if self.order == "sum-prod":
            ctx = self.prod_conv.sample(ctx)
            ctx = self.sum_layer.sample(ctx)
        elif self.order == "prod-sum":
            ctx = self.sum_layer.sample(ctx)
            ctx = self.prod_conv.sample(ctx)
        return ctx

    def _enable_input_cache(self):
        self.sum_layer._enable_input_cache()

    def _disable_input_cache(self):
        self.sum_layer._disable_input_cache()

    def __repr__(self):
        if self.order == "sum-prod":
            return f"SumProdLayer({self.sum_layer}, {self.prod_conv})"
        elif self.order == "prod-sum":
            return f"SumProdLayer({self.prod_conv}, {self.sum_layer})"
        else:
            raise ValueError(f"Order {self.order} not supported")


@dataclass(frozen=True)
class ConvPcConfig:
    """Class for the configuration of an Einet."""

    channels: List[int]  # Number of channels at each layer
    kernel_size: int  # Kernel size for the convolutional layers
    num_channels: int = 1  # Number of data input channels per feature
    num_classes: int = 1  # Number of root heads / Number of classes
    leaf_type: Type = None  # Type of the leaf base class (Normal, Bernoulli, etc)
    leaf_kwargs: Dict[str, Any] = field(default_factory=dict)  # Parameters for the leaf base class
    structure: str = "top-down"  # Structure of the Einet: top-down or bottom-up
    order: str = "sum-prod"  # Order of the sum and product layers

    def assert_valid(self):
        """Check whether the configuration is valid."""

        # Check that each dimension is valid
        check_valid(self.num_channels, int, 1)
        check_valid(self.num_classes, int, 1)
        assert self.leaf_type is not None, "EinetConfig.leaf_type parameter was not set!"
        assert self.order in [
            "sum-prod",
            "prod-sum",
        ], f"Invalid order type {self.order}. Must be 'sum-prod' or 'prod-sum'."
        assert self.structure in [
            "top-down",
            "bottom-up",
        ], f"Invalid structure type {self.structure}. Must be 'top-down' or 'bottom-up'."

        assert isinstance(self.leaf_type, type) and issubclass(
            self.leaf_type, AbstractLeaf
        ), f"Parameter EinetConfig.leaf_base_class must be a subclass type of Leaf but was {self.leaf_type}."


class ConvPc(nn.Module):
    def __init__(self, config: ConvPcConfig, data_shape: Shape):

        super().__init__()
        self.config = config
        self.data_shape = data_shape

        # Data leaf

        self.leaf_data = self.config.leaf_type(
            num_features=self.data_shape.num_pixels,
            num_channels=self.data_shape.channels,
            num_leaves=self.config.channels[0],
            num_repetitions=1,
            **self.config.leaf_kwargs,
        )

        # Construct layers
        layers = []
        h, w = self.data_shape.height, self.data_shape.width

        kh, kw = self.config.kernel_size, self.config.kernel_size

        if self.config.structure == "bottom-up":
            for i in range(0, len(self.config.channels) - 1):
                layers.append(
                    SumProdLayer(
                        in_channels=self.config.channels[i],
                        out_channels=self.config.channels[i + 1],
                        height=h,
                        width=w,
                        kernel_size_h=kh,
                        kernel_size_w=kw,
                        order=config.order,
                    )
                )

                # Update height and width according to the kernel size
                h, w = h // kh, w // kw

            # Reduce to a single dimension by setting the kernel size the the current height and width
            # Add a root sum layer to reduce to 1 channel (single root node)
            layers.append(
                SumProdLayer(
                    order="prod-sum",
                    in_channels=self.config.channels[-1],
                    out_channels=1,
                    height=h,
                    width=w,
                    kernel_size_h=h,
                    kernel_size_w=w,
                )
            )

        # TODO: Investigate top down approach as well
        elif self.config.structure == "top-down":

            # Add sum layer on top
            layers.append(SumLayer(in_channels=self.config.channels[-1], out_channels=1, height=1, width=1))

            # Build from top down
            h, w = 1, 1
            channels = [self.config.channels[0]] + self.config.channels
            for i in reversed(range(0, len(self.config.channels))):
                h, w = h * 2, w * 2
                layers.append(
                    SumProdLayer(
                        in_channels=channels[i],
                        out_channels=channels[i + 1],
                        height=h,
                        width=w,
                        kernel_size_h=2,
                        kernel_size_w=2,
                        order=config.order,
                    )
                )

            # Add lowest SumProd layer to reduce data height/width to h/w
            layers.append(
                ProdConv(
                    kernel_size_h=self.data_shape.height // h,
                    kernel_size_w=self.data_shape.width // w,
                )
            )
            layers = reversed(layers)

        else:
            raise ValueError(f"Split {self.config.structure} not supported")

        self.layers = nn.ModuleList(layers)

    @property
    def __device(self):
        """Small hack to obtain the current device."""
        return next(self.parameters()).device

    def _forward_leaf_data(self, x):
        x = x.view(x.size(0), self.data_shape.channels, self.data_shape.height * self.data_shape.width)
        x = self.leaf_data(x, marginalized_scopes=None)
        x = x.view(
            x.size(0), self.data_shape.channels, self.data_shape.height, self.data_shape.width, self.config.channels[0]
        )

        # Factorize data input channels
        x = x.sum(1)

        # Remove empty repetition dimension (artifact from Einet implementation)
        x = x.squeeze(-1)

        # Permute from (N, H, W, C) to (N, C, H, W)
        x = x.permute(0, 3, 1, 2)

        return x

    def forward(self, x: torch.Tensor | None = None):
        # Apply leaf distributions (replace marginalization indicators with 0.0 first)
        x = self._forward_leaf_data(x)

        for layer in self.layers:
            x = layer(x)

        return x.view(x.size(0))

    def sample(
        self,
        num_samples: int | None = None,
        evidence: torch.Tensor | None = None,
        class_index = None,
        seed: int | None = None,
        is_mpe: bool = False,
        mpe_at_leaves: bool = False,
        return_leaf_params: bool = False,
        temperature_leaves: float = 1.0,
        temperature_sums: float = 1.0,
        marginalized_scopes: List[int] | None = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:

        if num_samples is None and evidence is None:
            raise ValueError("Either num_samples or evidence must be given.")
        if num_samples is None and evidence is not None:
            num_samples = evidence.size(0)

        ctx = SamplingContext(
            num_samples=num_samples,
            mpe_at_leaves=mpe_at_leaves,
            is_differentiable=True,
            evidence=evidence,
            return_leaf_params=return_leaf_params,
            is_mpe=is_mpe,
            temperature_sums=temperature_sums,
            temperature_leaves=temperature_leaves,
        )

        # Init indices_out: (N, C=1, H=1, W=1)
        ctx.indices_out = torch.ones(
            size=(num_samples, 1, 1, 1), dtype=torch.float, device=self.__device, requires_grad=True
        )

        ctx.indices_repetition = torch.ones(
            size=(num_samples, 1), dtype=torch.float, device=self.__device, requires_grad=True
        )

        with sampling_context(
            self,
            evidence=evidence,
            requires_grad=True,
            seed=seed,
        ):

            # Iterate over layers
            for i, layer in reversed(list(enumerate(self.layers))):

                # Sample layer
                ctx = layer.sample(ctx=ctx)

            indices_out = ctx.indices_out
            ctx.indices_out = None
            samples = self.leaf_data.sample(ctx=ctx)

            # Samples are of shape (N, C, H*W, I)
            N, C, HW, I = samples.shape
            samples = samples.view(N, self.data_shape.channels, self.data_shape.height, self.data_shape.width, I)
            # (N, C, H, W, I) -> (N, I, C, H, W)
            samples = samples.permute(0, 4, 1, 2, 3)

            indices_out.unsqueeze_(2)  # Make space for in_channels (I) dim
            # Index into I
            samples = index_one_hot(samples, index=indices_out, dim=1)

        if evidence is not None:
            # First make a copy such that the original object is not changed
            evidence = evidence.clone().float()
            shape_evidence = evidence.shape
            evidence = evidence.view_as(samples)
            mask = torch.isnan(evidence)
            evidence[mask] = samples[mask].to(evidence.dtype)
            evidence = evidence.view(shape_evidence)

            return evidence
        else:
            return samples


if __name__ == "__main__":
    channels = [3, 6, 12, 24, 50, 20]

    from simple_einet.conv_pc import ConvPcConfig

    config = ConvPcConfig(
        channels=channels,
        kernel_size=2,
        num_channels=3,
        num_classes=1,
        leaf_type=Binomial,
        leaf_kwargs={"total_count": 255},
        # structure="top-down",
        structure="bottom-up",
        order="sum-prod",
    )

    data_shape = Shape(channels=3, height=32, width=32)

    model = ConvPc(config, data_shape)

    x = torch.randint(low=0, high=255, size=(2, 3, 32, 32)).float()

    print(model)
    print(model(x))
    print(model.sample(2))

    from simple_einet.mixture import Mixture
    mix = Mixture(n_components=10, config=config, data_shape=data_shape)
    mix.initialize(data=x)
    print(mix)
    print(mix(x))
    print(mix.sample(2))
