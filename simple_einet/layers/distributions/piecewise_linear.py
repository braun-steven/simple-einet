from typing import List, Tuple

import tqdm
import itertools
from collections import defaultdict
from pytorch_lightning.strategies import deepspeed
import torch
from torch import nn

from simple_einet.dist import DataType, Domain
from simple_einet.layers.distributions.abstract_leaf import AbstractLeaf
from simple_einet.sampling_utils import SamplingContext
from simple_einet.type_checks import check_valid
from simple_einet.histogram import _get_bin_edges_torch


import logging

from fast_pytorch_kmeans import KMeans

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
                                densities = torch.histogram(data_subset.cpu(), bins=breaks.cpu(), density=True).hist.to(data.device)


                        elif self.domains[i_feature].data_type == DataType.CONTINUOUS:
                            # Find histogram bins using numpys "auto" logic
                            bins, _ = _get_bin_edges_torch(data_subset)

                            # Construct histogram
                            densities = torch.histogram(data_subset.cpu(), bins=bins.cpu(), density=True).hist.to(data.device)
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
                        x = torch.tensor(x, device=data.device) #, requires_grad=True)
                        y = torch.tensor(y, device=data.device) #, requires_grad=True)

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

        # self.mixture_weights = torch.zeros(self.num_features, self.num_channels, self.num_repetitions, self.num_leaves, device=data.device)
        # for i_feature in range(self.num_features):
        #     xs_channel = []
        #     ys_channel = []
        #     for i_channel in range(self.num_channels):

        #         # Select relevant data
        #         data_subset = data[:, i_channel, i_feature].view(data.shape[0], 1).float()

        #         # Repeat this for every repetition
        #         xs_repetition = []
        #         ys_repetition = []
        #         for i_repetition in range(self.num_repetitions):

        #             # Cluster data into num_leaves clusters
        #             kmeans = KMeans(n_clusters=self.num_leaves, mode="euclidean", verbose=0, init_method="kmeans++")
        #             kmeans.fit(data_subset)

        #             predictions = kmeans.predict(data_subset.view(data_subset.shape[0], -1).float())
        #             counts = torch.bincount(predictions)
        #             self.mixture_weights[i_feature, i_channel, i_repetition] = counts / counts.sum()

        #             # Get cluster assigments for each datapoint
        #             cluster_idxs = kmeans.max_sim(a=data_subset, b=kmeans.centroids)[1]

        #             xs_leaves = []
        #             ys_leaves = []
        #             for cluster_idx in range(self.num_leaves):

        #                 # Select data for this cluster
        #                 mask = cluster_idxs == cluster_idx
        #                 cluster_data = data_subset[mask]

        #                 # Construct histogram
        #                 if self.domains[i_feature].data_type == DataType.DISCRETE:
        #                     # Edges are the discrete values
        #                     mids = torch.tensor(self.domains[i_feature].values, device=data.device).float()

        #                     # Add a break at the end
        #                     breaks = torch.cat([mids, torch.tensor([mids[-1] + 1])])

        #                     if cluster_data.shape[0] == 0:
        #                         # If there is no data in this cluster, set the density to uniform
        #                         densities = torch.ones(len(mids), device=data.device) / len(mids)
        #                     else:
        #                         # Compute counts
        #                         densities = torch.histogram(cluster_data, bins=breaks, density=True).hist


        #                 elif self.domains[i_feature].data_type == DataType.CONTINUOUS:
        #                     # Find histogram bins using numpys "auto" logic
        #                     bins, _ = _get_bin_edges_torch(cluster_data)

        #                     # Construct histogram
        #                     densities = torch.histogram(cluster_data, bins=bins, density=True).hist
        #                     breaks = bins
        #                     mids = ((breaks + torch.roll(breaks, shifts=-1, dims=0)) / 2)[:-1]
        #                 else:
        #                     raise ValueError(f"Unknown data type: {domains[i_feature]}")

        #                 # Apply optional laplace smoothing
        #                 if self.alpha > 0:
        #                     n_samples = cluster_data.shape[0]
        #                     n_bins = len(breaks) - 1
        #                     counts = densities * n_samples
        #                     alpha_abs = n_samples * self.alpha
        #                     densities = (counts + alpha_abs) / (n_samples + n_bins * alpha_abs)

        #                 assert len(densities) + 1 == len(breaks)

        #                 # Add tail breaks to start and end
        #                 if self.domains[i_feature].data_type == DataType.DISCRETE:
        #                     tail_width = 1
        #                     x = [b for b in breaks[:-1]]
        #                     x = [x[0] - tail_width] + x + [x[-1] + tail_width]
        #                 elif self.domains[i_feature].data_type == DataType.CONTINUOUS:
        #                     EPS = 1e-8
        #                     x = (
        #                         [breaks[0] - EPS]
        #                         + [b0 + (b1 - b0) / 2 for (b0, b1) in pairwise(breaks)]
        #                         + [breaks[-1] + EPS]
        #                     )
        #                 else:
        #                     raise ValueError(f"Unknown data type: {domains[i_feature].data_type}")

        #                 # Add density 0 at start an end tail break
        #                 y = [0.0] + [d for d in densities] + [0.0]

        #                 # Check that shapes still match
        #                 assert len(densities) == len(breaks) - 1
        #                 assert len(x) == len(y), (len(x), len(y))

        #                 # Construct tensors
        #                 x = torch.tensor(x, device=data.device) #, requires_grad=True)
        #                 y = torch.tensor(y, device=data.device) #, requires_grad=True)

        #                 # Compute AUC using the trapeziod rule
        #                 auc = torch.trapezoid(x=x, y=y)

        #                 # Normalize y to sum to 1 using AUC
        #                 y = y / auc

        #                 # Store
        #                 xs_leaves.append(x)
        #                 ys_leaves.append(y)

        #             xs_repetition.append(xs_leaves)
        #             ys_repetition.append(ys_leaves)

        #         # Store
        #         xs_channel.append(xs_repetition)
        #         ys_channel.append(ys_repetition)

        #     # Store
        #     xs.append(xs_channel)
        #     ys.append(ys_channel)

        # # Check shapes
        # assert len(xs) == len(ys) == self.num_features
        # assert len(xs[0]) == len(ys[0]) == self.num_channels
        # assert len(xs[0][0]) == len(ys[0][0]) == self.num_repetitions
        # assert len(xs[0][0][0]) == len(ys[0][0][0]) == self.num_leaves


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
            raise ValueError("PiecewiseLinear leaf layer has not been initialized yet. Call initialize(...) first to estimate to correct piecewise linear functions upfront.")
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
        # self.num_features = len(xs)
        # self.num_channels = len(xs[0])
        # self.num_repetitions = len(xs[0][0])
        # self.num_leaves = len(xs[0][0][0])

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
                        # xs_i = self.xs[i_feature][i_channel][i_repetition][i_leaf]
                        # ys_i = self.ys[i_feature][i_channel][i_repetition][i_leaf]
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
                        # xs_i = self.xs[i_feature][i_channel][i_repetition][i_leaf]
                        # ys_i = self.ys[i_feature][i_channel][i_repetition][i_leaf]
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
                        # xs_i = self.xs[i_feature][i_channel][i_repetition][i_leaf]
                        # ys_i = self.ys[i_feature][i_channel][i_repetition][i_leaf]
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


if __name__ == "__main__":
    # # Test the piecewise linear distribution
    # data = torch.randn(1000, 3, 30)
    # data_types = [DataType.CONTINUOUS] * 30
    # pl = PiecewiseLinear(num_features=30, num_channels=3, num_leaves=7, num_repetitions=3)
    # pl.initialize(data, data_types)

    # # Test the piecewise linear distribution
    # d = pl._get_base_distribution()
    # ll = d.log_prob(data)
    # samples = d.sample((10,))
    # mpes = d.mpe(10)
    # print(ll.shape)
    # print(samples.shape)
    # print(mpes.shape)

    # from simple_einet.einet import Einet, EinetConfig

    # # Create an Einet
    # einet = Einet(
    #     EinetConfig(depth=3, num_features=8, num_channels=1, num_leaves=3, num_repetitions=4, leaf_type=PiecewiseLinear)
    # )

    # # Create some data
    # data = torch.randn(1000, 1, 8)

    # einet.leaf.base_leaf.initialize(data, [DataType.CONTINUOUS] * 8)

    # # Test the piecewise linear distribution
    # einet.sample(num_samples=10)
    # einet.mpe()

    # exit(0)
    import seaborn as sns

    sns.set()
    sns.set_style("whitegrid")

    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from torch.distributions import Normal, Uniform

    # Create a multimodal 1D dataset
    def create_multimodal_dataset(n_samples=100000):
        # Mix of three Gaussian distributions and one Uniform distribution
        dist1 = Normal(loc=-5, scale=1)
        dist2 = Normal(loc=0, scale=0.5)
        dist3 = Normal(loc=5, scale=1.5)
        dist4 = Uniform(low=-2, high=2)

        # Generate samples
        samples1 = dist1.sample((int(n_samples * 0.3),))
        samples2 = dist2.sample((int(n_samples * 0.2),))
        samples3 = dist3.sample((int(n_samples * 0.3),))
        samples4 = dist4.sample((int(n_samples * 0.2),))

        # Combine samples
        all_samples = torch.cat([samples1, samples2, samples3, samples4])

        # Shuffle the samples
        return all_samples[torch.randperm(all_samples.size(0))]

    # Create the dataset
    data = create_multimodal_dataset().unsqueeze(1).unsqueeze(1)  # Shape: (10000, 1, 1)

    # Initialize PiecewiseLinear
    num_features = 1
    num_channels = 1
    num_leaves = 1  # You mentioned this was increased for flexibility, but it's still 1 here
    num_repetitions = 1

    pl = PiecewiseLinear(
        num_features=num_features, num_channels=num_channels, num_leaves=num_leaves, num_repetitions=num_repetitions
    )
    pl.initialize(data, [Domain.continuous_inf_support()])

    # Get the base distribution
    d = pl._get_base_distribution()

    # Calculate log probabilities for a range of values
    x_range = torch.linspace(-10, 10, 100000).unsqueeze(1).unsqueeze(1)
    log_probs = d.log_prob(x_range)

    # Generate samples from the PWL distribution
    pwl_samples = d.sample((100000,)).squeeze()

    # Plot the results
    plt.figure(figsize=(12, 6))

    # Plot histogram of the original data
    plt.hist(data.squeeze().numpy(), bins=100, density=True, alpha=0.5, label="Original Data")

    # Plot histogram of the PWL samples
    plt.hist(pwl_samples.numpy(), bins=100, density=True, alpha=0.5, label="PWL Samples")

    # Plot the PWL log probability (exponentiated for density)
    plt.plot(x_range.squeeze().numpy(), torch.exp(log_probs).squeeze().numpy(), "r-", linewidth=2, label="PWL Density")

    pwl_mpe_x = d.mpe(1)
    # Plot MPE of the distribution at the y position of the pwl_mpe value
    pwl_mpe_y = d.log_prob(pwl_mpe_x).exp()
    plt.plot(pwl_mpe_x.squeeze(), pwl_mpe_y.squeeze(), "rx", markersize=13, label="PWL MPE")

    plt.title("Multimodal Data, PWL Distribution, and PWL Samples")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("/tmp/continuous_pwl.png", dpi=300)

    # Print some statistics
    print(f"Log probability shape: {d.log_prob(data).shape}")
    print(f"Sample shape: {d.sample((10,)).shape}")
    print(f"MPE shape: {d.mpe(10).shape}")

    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from torch.distributions import Categorical

    # Create a multimodal discrete dataset
    def create_discrete_multimodal_dataset(n_samples=100000):
        # Define probabilities for a multimodal discrete distribution
        probs = torch.tensor([1, 2.8, 6, 3, 0.5, 0.7, 2, 3.5, 5, 7, 8, 4, 3, 2, 2, 1, 0.5])
        probs = probs / probs.sum()
        num_categories = len(probs)

        # Create a Categorical distribution
        dist = Categorical(probs)

        # Generate samples
        samples = dist.sample((n_samples,))

        return samples, num_categories

    # Create the dataset
    data, num_categories = create_discrete_multimodal_dataset()
    data = data.unsqueeze(1).unsqueeze(1)  # Shape: (10000, 1, 1)

    # Initialize PiecewiseLinear
    num_features = 1
    num_channels = 1
    num_leaves = 1  # You mentioned this should be set to the number of categories, but it's still 1 here
    num_repetitions = 1

    pl = PiecewiseLinear(
        num_features=num_features, num_channels=num_channels, num_leaves=num_leaves, num_repetitions=num_repetitions
    )
    pl.initialize(data, [Domain.discrete_range(min=0, max=num_categories)])

    # Get the base distribution
    d = pl._get_base_distribution()

    # Calculate probabilities for a range of values (including fractional values)
    x_range = torch.linspace(-0.5, num_categories - 0.5, 100000).unsqueeze(1).unsqueeze(1)
    log_probs = d.log_prob(x_range)
    probs = torch.exp(log_probs)

    # Generate samples from the PWL distribution
    pwl_samples = d.sample((1000000,)).squeeze()

    # Plot the results
    plt.figure(figsize=(12, 6))

    # Plot histogram of the original data
    plt.hist(
        data.squeeze().numpy(), bins=np.arange(num_categories + 1) - 0.5, density=True, alpha=0.5, label="Original Data"
    )

    # Plot histogram of the PWL samples
    plt.hist(
        pwl_samples.numpy(), bins=np.arange(num_categories + 1) - 0.5, density=True, alpha=0.5, label="PWL Samples"
    )

    # Plot the PWL probabilities as a line
    plt.plot(x_range.squeeze().numpy(), probs.squeeze().numpy(), "r-", linewidth=2, label="PWL Distribution")

    # Plot MPE of the distribution at the y position of the pwl_mpe value
    pwl_mpe_x = d.mpe(1)
    pwl_mpe_y = d.log_prob(pwl_mpe_x).exp()
    plt.plot(pwl_mpe_x.squeeze(), pwl_mpe_y.squeeze(), "rx", markersize=13, label="PWL MPE")

    plt.title("Discrete Multimodal Data, PWL Distribution, and PWL Samples")
    plt.xlabel("Value")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(range(num_categories))
    plt.xlim(-0.5, num_categories - 0.5)
    plt.savefig("/tmp/discrete_pwl.png", dpi=300)

    # Print some statistics
    print(f"Log probability shape: {d.log_prob(data).shape}")
    print(f"Sample shape: {d.sample((10,)).shape}")
    print(f"MPE shape: {d.mpe(10).shape}")

    # Calculate and print the actual probabilities
    actual_probs = torch.bincount(data.squeeze().long(), minlength=num_categories).float() / len(data)
    print("\nActual probabilities:")
    print(actual_probs)

    print("\nPWL probabilities at integer points:")
    pwl_probs_at_integers = torch.exp(d.log_prob(torch.arange(num_categories).float().unsqueeze(1).unsqueeze(1)))
    print(pwl_probs_at_integers.squeeze())

    # Calculate KL divergence
    kl_div = torch.sum(actual_probs * torch.log(actual_probs / pwl_probs_at_integers.squeeze()))
    print(f"\nKL Divergence: {kl_div.item():.4f}")
