from typing import List, Tuple

import numpy as np
import torch
from torch import distributions as dist
from torch import nn
from torch.nn import functional as F

from simple_einet.layers.distributions.abstract_leaf import AbstractLeaf, dist_mode, dist_forward
from simple_einet.sampling_utils import SamplingContext
from simple_einet.type_checks import check_valid

from icecream import ic


class MultivariateNormal(AbstractLeaf):
    """Multivariate Gaussian layer."""

    def __init__(
        self,
        num_features: int,
        num_channels: int,
        num_leaves: int,
        cardinality: int,
        num_repetitions: int = 1,
    ):
        """Creat a multivariate gaussian layer.

        # NOTE: Marginalization in this leaf is not supported for now. Reason: The implementation parallelizes the
        # representation of the different multivariate gaussians for efficiency. That means we have num_dists x (K x K)
        # gaussians of cardinality K. If we were now to marginalize a single scope, we would need to pick the distributions
        # in which that scope appears and reduce these gaussians to size (K-1 x K-1) by deleting the appropriate elements
        # in their means and rows/columns in their covariance matrices. This results in non-parallelizable execution of
        # computations with the other non-marginalized gaussians.

        Args:
            num_features: Number of input features.
            num_channels: Number of input channels.
            num_leaves: Number of parallel representations for each input feature.
            num_repetitions: Number of parallel repetitions of this layer.
            cardinality: Number of features covered.

        """
        super().__init__(num_features, num_channels, num_leaves, num_repetitions, cardinality)
        self._pad_value = num_features % cardinality
        self.out_features = np.ceil(num_features / cardinality).astype(int)

        # Number of separate mulitvariate normal distributions, each covering #cardinality features
        param_shape = (num_channels, self.out_features, num_leaves, num_repetitions)
        self._num_dists = np.prod(param_shape)

        # Create gaussian means and covs
        self.means = nn.Parameter(torch.randn(*param_shape, cardinality))

        # Generate covariance matrix via the cholesky decomposition: s = L'L where L is a triangular matrix
        self.L_diag_log = nn.Parameter(torch.zeros(*param_shape, cardinality))
        self.L_offdiag = nn.Parameter(torch.tril(torch.zeros(*param_shape, cardinality, cardinality), diagonal=-1))

    @property
    def scale_tril(self):
        """Get the lower triangular matrix L of the cholesky decomposition of the covariance matrix."""
        L_diag = self.L_diag_log.exp()  # Ensure that L_diag is positive
        L_offdiag = self.L_offdiag.tril(-1)  # Take the off-diagonal part of L_offdiag
        L_full = torch.diag_embed(L_diag) + L_offdiag  # Construct full lower triangular matrix
        return L_full

    def _get_base_distribution(self, ctx: SamplingContext = None, marginalized_scopes = None):
        # View means and scale_tril
        means = self.means.view(self._num_dists, cardinality)
        scale_tril = self.scale_tril.view(self._num_dists, cardinality, cardinality)


        mv = CustomMultivariateNormalDist(
            mean=means,
            scale_tril=scale_tril,
            num_channels=self.num_channels,
            out_features=self.out_features,
            num_leaves=self.num_leaves,
            num_repetitions=self.num_repetitions,
            cardinality=self.cardinality,
            pad_value=self._pad_value,
            num_dists=self._num_dists,
            num_features=self.num_features,
        )
        return mv


class CustomMultivariateNormalDist:
    def __init__(
        self,
        mean,
        scale_tril,
        num_dists,
        num_channels,
        out_features,
        num_leaves,
        num_repetitions,
        cardinality,
        pad_value,
        num_features,
    ):
        self.mean = mean
        self.scale_tril = scale_tril
        self.num_channels = num_channels
        self.out_features = out_features
        self.num_leaves = num_leaves
        self.num_features = num_features
        self.num_repetitions = num_repetitions
        self.cardinality = cardinality
        self._pad_value = pad_value
        self._num_dists = num_dists

        self.mv = dist.MultivariateNormal(loc=self.mean, scale_tril=self.scale_tril)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        # Pad dummy variable via reflection
        if self._pad_value != 0:
            x = F.pad(x, pad=[0, 0, 0, self._pad_value], mode="reflect")

        # Split features into groups
        x = x.view(
            batch_size,
            self.num_channels,
            self.out_features,
            1,
            1,
            self.cardinality,
        )

        # Repeat groups by number of output_channels and number of repetitions
        x = x.repeat(1, 1, 1, self.num_leaves, self.num_repetitions, 1)

        # Merge groups and repetitions
        x = x.view(batch_size, self._num_dists, self.cardinality)

        # Compute multivariate gaussians
        x = self.mv.log_prob(x)
        x = x.view(x.shape[0], self.num_channels, self.out_features, self.num_leaves, self.num_repetitions)
        return x

    def sample(self, sample_shape: Tuple[int]) -> torch.Tensor:
        samples = self.mv.sample(sample_shape=sample_shape)

        samples = samples.view(
            sample_shape[0],
            self.num_channels,
            self.out_features,
            self.num_leaves,
            self.num_repetitions,
            self.cardinality,
        )

        samples = samples.permute(0, 1, 2, 5, 3, 4)
        samples = samples.reshape(
            sample_shape[0], self.num_channels, self.num_features, self.num_leaves, self.num_repetitions
        )

        # Add empty dimension to align shape with samples from uniform distributions such that this layer can be used in
        # the same way as other layers in dist_sample (abstract_leaf.py)
        samples.unsqueeze_(1)

        return samples

    def mpe(self, num_samples) -> torch.Tensor:
        """
        Generates MPE samples from this distribution.

        Args:
            num_samples (int): The number of samples to generate.

        Returns:
            samples (torch.Tensor):
        """
        samples = self.mean.repeat(num_samples, 1, 1, 1, 1)

        samples = samples.view(
            num_samples,
            self.num_channels,
            self.out_features,
            self.num_leaves,
            self.num_repetitions,
            self.cardinality,
        )

        samples = samples.permute(0, 1, 2, 5, 3, 4)
        samples = samples.reshape(
            num_samples, self.num_channels, self.num_features, self.num_leaves, self.num_repetitions
        )
        return samples


if __name__ == "__main__":
    # The following code is a test snippet that generates multiple 2D gaussian distributions, fits a multivariate normal distribution and visualizes the data against the fitted distribution.

    # Import necessary modules
    # Torch for the model and optimization, numpy for data manipulation, matplotlib for plotting
    import torch
    import torch.nn as nn
    import torch.distributions as dist
    import torch.optim as optim
    import numpy as np
    from typing import List
    import matplotlib.pyplot as plt

    torch.manual_seed(1)
    np.random.seed(1)

    import seaborn as sns

    # Apply seaborn's default style to make plots more aesthetically pleasing
    sns.set_style("whitegrid")

    # Function to generate synthetic 2D data from two multivariate Gaussian distributions
    # This serves as the dataset for which we want to fit a multivariate normal distribution
    def generate_data(num_samples=100):
        # Parameters for first Gaussian blob
        mean1 = [2.0, 3.0]
        cov1 = [[1.0, 0.9], [0.9, 0.5]]

        # Parameters for second Gaussian blob
        mean2 = [-1.0, -2.0]
        cov2 = [[0.4, -0.1], [-0.1, 0.3]]

        # Parameters for third Gaussian blob
        mean3 = [4.0, -1.0]
        cov3 = [[0.3, 0.2], [0.2, 0.5]]

        # Parameters for fourth Gaussian blob
        mean4 = [-3.0, 2.0]
        cov4 = [[0.5, -0.2], [-0.2, 0.3]]

        # Generate data points
        data1 = np.random.multivariate_normal(mean1, cov1, num_samples // 4)
        data2 = np.random.multivariate_normal(mean2, cov2, num_samples // 4)
        data3 = np.random.multivariate_normal(mean3, cov3, num_samples // 4)
        data4 = np.random.multivariate_normal(mean4, cov4, num_samples // 4)
        data = np.vstack([data1, data2, data3, data4])

        return torch.tensor(data, dtype=torch.float32)

    # Function to plot both the generated data and the learned probability density
    def plot_data_and_distribution_seaborn(data, samples, model):
        sns.set(style="whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

        # Generate a grid over which we evaluate the model's density function
        x, y = np.linspace(data[:, 0].min(), data[:, 0].max(), 100), np.linspace(
            data[:, 1].min(), data[:, 1].max(), 100
        )
        X, Y = np.meshgrid(x, y)
        grid = np.vstack([X.ravel(), Y.ravel()]).T

        # Evaluate the learned density function over the grid
        with torch.no_grad():
            grid_tensor = torch.tensor(grid, dtype=torch.float32)
            log_prob = model(grid_tensor)
            prob_density = log_prob.exp().numpy().ravel()  # Ensure this is 1-dimensional

        # Plot for original data points using Seaborn
        sns.scatterplot(x=data[:, 0], y=data[:, 1], ax=axes[0], color="green", alpha=0.6, label="Original Data")
        sns.kdeplot(x=grid[:, 0], y=grid[:, 1], weights=prob_density, fill=True, ax=axes[0], cmap="viridis", alpha=0.5)
        axes[0].set_title("Original Data and Fitted Density")
        axes[0].legend()

        # Plot for sampled data points using Seaborn
        sns.scatterplot(x=samples[:, 0], y=samples[:, 1], ax=axes[1], color="blue", alpha=0.6, label="Sampled Data")
        sns.kdeplot(x=grid[:, 0], y=grid[:, 1], weights=prob_density, fill=True, ax=axes[1], cmap="plasma", alpha=0.5)
        axes[1].set_title("Samples and Fitted Density")
        axes[1].legend()

        plt.tight_layout()
        plt.show(dpi=120)

    # Generate synthetic 2D data
    from sklearn.datasets import make_moons

    n_samples = 400
    data = generate_data(n_samples)
    # data = torch.tensor(make_moons(n_samples=n_samples, noise=0.1, random_state=0)[0])

    # Initialize the Multivariate Normal model
    # The model will be trained to fit the synthetic data
    num_features = 2
    num_channels = 1
    num_leaves = 4
    num_repetitions = 1
    cardinality = 2

    from simple_einet.einet import Einet, EinetConfig

    cfg = EinetConfig(
        num_features=num_features,
        num_channels=num_channels,
        num_leaves=num_leaves,
        depth=0,
        num_repetitions=num_repetitions,
        num_classes=1,
        leaf_type=MultivariateNormal,
        leaf_kwargs={"cardinality": cardinality},
    )
    model = Einet(cfg)

    # Setup optimization
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    epochs = 1000

    # Training loop to fit the Multivariate Normal model
    for epoch in range(epochs):
        optimizer.zero_grad()
        log_prob = model(data)

        # Negative log-likelihood as loss function
        loss = -torch.mean(log_prob)
        loss.backward()
        optimizer.step()

        # Logging to monitor progress
        if epoch % 50 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}")

    # Sample
    samples = model.sample(num_samples=n_samples)
    samples.squeeze_(1)
    ic(samples.shape)

    plot_data_and_distribution_seaborn(data, samples, model)
