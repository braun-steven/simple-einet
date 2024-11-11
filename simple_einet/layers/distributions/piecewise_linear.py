import itertools
import logging
from collections import defaultdict
from typing import List, Tuple

import torch
import tqdm
from fast_pytorch_kmeans import KMeans
from simple_einet.dist import DataType, Domain
from simple_einet.histogram import _get_bin_edges_torch
from simple_einet.layers.distributions.abstract_leaf import AbstractLeaf
from simple_einet.sampling_utils import SamplingContext
from simple_einet.type_checks import check_valid
from torch import nn

logger = logging.getLogger(__name__)


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


class PiecewiseLinear(AbstractLeaf):
    """
    Piecewise linear leaf implementation.

    First constructs a histogram from the data and then approximates the histogram with a piecewise linear function.
    """

    def __init__(
        self,
        num_features: int,
        num_channels: int,
        num_leaves: int,
        num_repetitions: int,
        alpha: float = 0.0,
    ):
        """
        Initializes a piecewise linear distribution with the given parameters.

        Args:
            num_features (int): The number of features in the input tensor.
            num_channels (int): The number of channels in the input tensor.
            num_leaves (int): The number of leaves in the tree structure.
            num_repetitions (int): The number of repetitions of the tree structure.
            alpha (float): The alpha parameter for optional laplace smoothing.
        """
        super().__init__(num_features, num_channels, num_leaves, num_repetitions)
        self.alpha = check_valid(alpha, expected_type=float, lower_bound=0.0)
        self.xs = None
        self.ys = None
        self.is_initialized = False  # Flag to check if the distribution has been initialized

    def initialize(self, data: torch.Tensor, domains: List[Domain]):
        """
        Initializes the piecewise linear distribution with the given data.

        This includes the following steps:

        1. Cluster data into num_leaves clusters, num_repetition times, such that for each leaf representation and repetition there is a piecewise linear function.
        2. For each cluster, construct a histogram of the data.
        3. Approximate the histogram with a piecewise linear function.

        Args:
            data (torch.Tensor): The data to initialize the distribution with.
            domains (List[Domain]): The domains of the features.
        """

        logger.info(f"Initializing piecewise linear distribution with data of shape {data.shape}.")

        assert data.shape[1] == self.num_channels
        assert data.shape[2] == self.num_features

        self.domains = domains

        # Parameters
        xs = []  # [R, L, F, C]
        ys = []

        self.mixture_weights = torch.zeros(self.num_repetitions, self.num_leaves, device=data.device)

        for i_repetition in tqdm.tqdm(range(self.num_repetitions), desc="Initializing PiecewiseLinear Leaf Layer"):
            # Repeat this for every repetition
            xs_leaves = []
            ys_leaves = []

            # Cluster data into num_leaves clusters
            kmeans = KMeans(n_clusters=self.num_leaves, mode="euclidean", verbose=0, init_method="random")
            kmeans.fit(data.view(data.shape[0], -1).float())

            predictions = kmeans.predict(data.view(data.shape[0], -1).float())
            counts = torch.bincount(predictions)
            self.mixture_weights[i_repetition] = counts / counts.sum()

            # Get cluster assigments for each datapoint
            cluster_idxs = kmeans.max_sim(a=data.view(data.shape[0], -1).float(), b=kmeans.centroids)[1]
            for cluster_idx in range(self.num_leaves):

                # Select data for this cluster
                mask = cluster_idxs == cluster_idx
                cluster_data = data[mask]

                xs_features = []
                ys_features = []
                for i_feature in range(self.num_features):
                    xs_channels = []
                    ys_channels = []

                    for i_channel in range(self.num_channels):

                        # Select relevant data
                        data_subset = cluster_data[:, i_channel, i_feature].view(cluster_data.shape[0], 1).float()

                        # Construct histogram
                        if self.domains[i_feature].data_type == DataType.DISCRETE:
                            # Edges are the discrete values
                            mids = torch.tensor(self.domains[i_feature].values, device=data.device).float()

                            # Add a break at the end
                            breaks = torch.cat([mids, torch.tensor([mids[-1] + 1], device=mids.device)])

                            if data_subset.shape[0] == 0:
                                # If there is no data in this cluster, set the density to uniform
                                densities = torch.ones(len(mids), device=data.device) / len(mids)
                            else:
                                # Compute counts
                                densities = torch.histogram(data_subset.cpu(), bins=breaks.cpu(), density=True).hist.to(
                                    data.device
                                )

                        elif self.domains[i_feature].data_type == DataType.CONTINUOUS:
                            # Find histogram bins using numpys "auto" logic
                            bins, _ = _get_bin_edges_torch(data_subset)

                            # Construct histogram
                            densities = torch.histogram(data_subset.cpu(), bins=bins.cpu(), density=True).hist.to(
                                data.device
                            )
                            breaks = bins
                            mids = ((breaks + torch.roll(breaks, shifts=-1, dims=0)) / 2)[:-1]
                        else:
                            raise ValueError(f"Unknown data type: {domains[i_feature]}")

                        # Apply optional laplace smoothing
                        if self.alpha > 0:
                            n_samples = data_subset.shape[0]
                            n_bins = len(breaks) - 1
                            counts = densities * n_samples
                            densities = (counts + self.alpha) / (n_samples + n_bins * self.alpha)

                        assert len(densities) + 1 == len(breaks)

                        # Add tail breaks to start and end
                        if self.domains[i_feature].data_type == DataType.DISCRETE:
                            tail_width = 1
                            x = [b for b in breaks[:-1]]
                            x = [x[0] - tail_width] + x + [x[-1] + tail_width]
                        elif self.domains[i_feature].data_type == DataType.CONTINUOUS:
                            EPS = 1e-8
                            x = (
                                [breaks[0] - EPS]
                                + [b0 + (b1 - b0) / 2 for (b0, b1) in pairwise(breaks)]
                                + [breaks[-1] + EPS]
                            )
                        else:
                            raise ValueError(f"Unknown data type: {domains[i_feature].data_type}")

                        # Add density 0 at start an end tail break
                        y = [0.0] + [d for d in densities] + [0.0]

                        # Check that shapes still match
                        assert len(densities) == len(breaks) - 1
                        assert len(x) == len(y), (len(x), len(y))

                        # Construct tensors
                        x = torch.tensor(x, device=data.device)  # , requires_grad=True)
                        y = torch.tensor(y, device=data.device)  # , requires_grad=True)

                        # Compute AUC using the trapeziod rule
                        auc = torch.trapezoid(x=x, y=y)

                        # Normalize y to sum to 1 using AUC
                        y = y / auc

                        # Store
                        xs_channels.append(x)
                        ys_channels.append(y)

                    # Store
                    xs_features.append(xs_channels)
                    ys_features.append(ys_channels)

                xs_leaves.append(xs_features)
                ys_leaves.append(ys_features)

            # Store
            xs.append(xs_leaves)
            ys.append(ys_leaves)

        # Check shapes
        assert len(xs) == len(ys) == self.num_repetitions
        assert len(xs[0]) == len(ys[0]) == self.num_leaves
        assert len(xs[0][0]) == len(ys[0][0]) == self.num_features
        assert len(xs[0][0][0]) == len(ys[0][0][0]) == self.num_channels

        # Store
        self.xs = xs
        self.ys = ys
        self.is_initialized = True

    def reset(self):
        self.is_initialized = False
        self.xs = None
        self.ys = None

    def _get_base_distribution(self, ctx: SamplingContext = None) -> "PiecewiseLinearDist":
        # Use custom normal instead of PyTorch distribution
        if not self.is_initialized:
            raise ValueError(
                "PiecewiseLinear leaf layer has not been initialized yet. Call initialize(...) first to estimate to correct piecewise linear functions upfront."
            )
        return PiecewiseLinearDist(self.xs, self.ys, domains=self.domains)

    def get_params(self):
        # Get params cannot be called on PiecewiseLinearDist, since it does not have any params
        raise NotImplementedError


def interp(
    x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor, dim: int = -1, extrapolate: str = "constant"
) -> torch.Tensor:
    """One-dimensional linear interpolation between monotonically increasing sample
    points, with extrapolation beyond sample points.

    Source: https://github.com/pytorch/pytorch/issues/50334#issuecomment-2304751532

    Returns the one-dimensional piecewise linear interpolant to a function with
    given discrete data points :math:`(xp, fp)`, evaluated at :math:`x`.

    Args:
        x: The :math:`x`-coordinates at which to evaluate the interpolated
            values.
        xp: The :math:`x`-coordinates of the data points, must be increasing.
        fp: The :math:`y`-coordinates of the data points, same shape as `xp`.
        dim: Dimension across which to interpolate.
        extrapolate: How to handle values outside the range of `xp`. Options are:
            - 'linear': Extrapolate linearly beyond range of xp values.
            - 'constant': Use the boundary value of `fp` for `x` values outside `xp`.

    Returns:
        The interpolated values, same size as `x`.
    """
    # Move the interpolation dimension to the last axis
    x = x.movedim(dim, -1)
    xp = xp.movedim(dim, -1)
    fp = fp.movedim(dim, -1)

    m = torch.diff(fp) / torch.diff(xp)  # slope
    b = fp[..., :-1] - m * xp[..., :-1]  # offset
    indices = torch.searchsorted(xp, x, right=False)

    if extrapolate == "constant":
        # Pad m and b to get constant values outside of xp range
        m = torch.cat([torch.zeros_like(m)[..., :1], m, torch.zeros_like(m)[..., :1]], dim=-1)
        b = torch.cat([fp[..., :1], b, fp[..., -1:]], dim=-1)
    else:  # extrapolate == 'linear'
        indices = torch.clamp(indices - 1, 0, m.shape[-1] - 1)

    values = m.gather(-1, indices) * x + b.gather(-1, indices)

    values = values.clamp(min=0.0)

    return values.movedim(-1, dim)


class PiecewiseLinearDist:
    def __init__(self, xs, ys, domains: list[DataType]):
        self.xs = xs
        self.ys = ys

        self.num_repetitions = len(xs)
        self.num_leaves = len(xs[0])
        self.num_features = len(xs[0][0])
        self.num_channels = len(xs[0][0][0])
        self.domains = domains

    def _compute_cdf(self, xs, ys):
        """Compute the CDF for the given piecewise linear function."""
        # Compute the integral over each interval using the trapezoid rule
        intervals = torch.diff(xs)
        trapezoids = 0.5 * intervals * (ys[:-1] + ys[1:])  # Partial areas

        # Cumulative sum to build the CDF
        cdf = torch.cat([torch.zeros(1, device=xs.device), torch.cumsum(trapezoids, dim=0)])

        # Normalize the CDF to ensure it goes from 0 to 1
        cdf = cdf / cdf[-1]

        return cdf

    def sample(self, sample_shape: torch.Size) -> torch.Tensor:
        """Sample from the piecewise linear distribution."""
        samples = torch.empty(
            (sample_shape[0], self.num_channels, self.num_features, self.num_leaves, self.num_repetitions),
            device=self.xs[0][0][0][0].device,
        )

        for i_feature in range(self.num_features):
            for i_channel in range(self.num_channels):
                for i_repetition in range(self.num_repetitions):
                    for i_leaf in range(self.num_leaves):
                        xs_i = self.xs[i_repetition][i_leaf][i_feature][i_channel]
                        ys_i = self.ys[i_repetition][i_leaf][i_feature][i_channel]

                        if self.domains[i_feature].data_type == DataType.DISCRETE:
                            # Sample from a categorical distribution
                            ys_i_wo_tails = ys_i[1:-1]  # Cut off the tail breaks
                            dist = torch.distributions.Categorical(probs=ys_i_wo_tails)
                            samples[..., i_channel, i_feature, i_leaf, i_repetition] = dist.sample(sample_shape)
                        elif self.domains[i_feature].data_type == DataType.CONTINUOUS:
                            # Compute the CDF for this piecewise function
                            cdf = self._compute_cdf(xs_i, ys_i)

                            # Sample from a uniform distribution
                            u = torch.rand(sample_shape, device=xs_i.device)

                            # Find the corresponding segment using searchsorted
                            indices = torch.searchsorted(cdf, u, right=True)

                            # Clamp indices to be within valid range
                            indices = torch.clamp(indices, 1, len(xs_i) - 1)

                            # Perform linear interpolation to get the sample value
                            x0, x1 = xs_i[indices - 1], xs_i[indices]
                            cdf0, cdf1 = cdf[indices - 1], cdf[indices]
                            slope = (x1 - x0) / (cdf1 - cdf0 + 1e-8)  # Avoid division by zero

                            # Compute the sampled value
                            samples[..., i_channel, i_feature, i_leaf, i_repetition] = x0 + slope * (u - cdf0)
                        else:
                            raise ValueError(f"Unknown data type: {self.domains[i_feature].data_type}")

        samples = samples.unsqueeze(
            1
        )  # Insert "empty" second dimension since all other distributions are implemented this way and the distribution sampling logic expects this

        return samples

    def mpe(self, num_samples: int) -> torch.Tensor:
        """Compute the most probable explanation (MPE) by taking the mode of the distribution."""
        modes = torch.empty(
            (num_samples, self.num_channels, self.num_features, self.num_leaves, self.num_repetitions),
            device=self.xs[0][0][0][0].device,
        )

        for i_feature in range(self.num_features):
            for i_channel in range(self.num_channels):
                for i_repetition in range(self.num_repetitions):
                    for i_leaf in range(self.num_leaves):
                        xs_i = self.xs[i_repetition][i_leaf][i_feature][i_channel]
                        ys_i = self.ys[i_repetition][i_leaf][i_feature][i_channel]

                        # Find the mode (the x value with the highest PDF value)
                        max_idx = torch.argmax(ys_i)
                        mode_value = xs_i[max_idx]

                        # Store the mode value
                        modes[:, i_channel, i_feature, i_leaf, i_repetition] = mode_value

        return modes

    def log_prob(self, x: torch.Tensor):
        # Initialize probs with ones of the same shape as obs
        probs = torch.zeros(list(x.shape[0:3]) + [self.num_leaves, self.num_repetitions], device=x.device)
        if x.dim() == 5:
            x = x.squeeze(-1).squeeze(-1)

        # Perform linear interpolation (equivalent to np.interp)
        for i_feature in range(self.num_features):
            for i_channel in range(self.num_channels):
                for i_repetition in range(self.num_repetitions):
                    for i_leaf in range(self.num_leaves):
                        xs_i = self.xs[i_repetition][i_leaf][i_feature][i_channel]
                        ys_i = self.ys[i_repetition][i_leaf][i_feature][i_channel]
                        ivalues = interp(x[:, i_channel, i_feature], xs_i, ys_i)
                        probs[:, i_channel, i_feature, i_leaf, i_repetition] = ivalues

        # Return the logarithm of probabilities
        logprobs = torch.log(probs)
        logprobs[logprobs == float("-inf")] = -300.0
        return logprobs

    def get_params(self):
        raise NotImplementedError

