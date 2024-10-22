from enum import Enum
import numpy as np


class Dist(str, Enum):
    """Enum for the distribution of the data."""

    NORMAL = "normal"
    MULTIVARIATE_NORMAL = "multivariate_normal"
    NORMAL_RAT = "normal_rat"
    BINOMIAL = "binomial"
    CATEGORICAL = "categorical"
    PIECEWISE_LINEAR = "piecewise_linear"


class DataType(str, Enum):
    """Enum for the type of the data."""

    CONTINUOUS = "continuous"
    DISCRETE = "discrete"

class Domain:
    def __init__(self, values=None, min=None, max=None, data_type=None):
        self.values = values
        self.min = min
        self.max = max
        self.data_type = data_type

    @staticmethod
    def discrete_bins(values):
        return Domain(min=min(values), max=max(values), values=values, data_type=DataType.DISCRETE)

    @staticmethod
    def discrete_range(min, max):
        return Domain(min=min, max=max, values=list(np.arange(min, max+1)), data_type=DataType.DISCRETE)

    @staticmethod
    def continuous_range(min, max):
        return Domain(min=min, max=max, data_type=DataType.CONTINUOUS)

    @staticmethod
    def continuous_inf_support():
        return Domain(min=np.NINF, max=np.inf, data_type=DataType.CONTINUOUS)



def get_data_type_from_dist(dist: Dist) -> DataType:
    """
    Returns the data type based on the distribution.

    Args:
        dist: The distribution.

    Returns:
        DataType: CONTINUOUS or DISCRETE based on the distribution.
    """
    if dist in {Dist.NORMAL, Dist.NORMAL_RAT, Dist.MULTIVARIATE_NORMAL, Dist.PIECEWISE_LINEAR}:
        return DataType.CONTINUOUS
    elif dist in {Dist.BINOMIAL, Dist.CATEGORICAL}:
        return DataType.DISCRETE
    else:
        raise ValueError(f"Unknown distribution ({dist}).")


def get_distribution(dist: Dist, cfg):
    """
    Get the distribution for the leaves.

    Args:
        dist: The distribution to use.

    Returns:
        leaf_type: The type of the leaves.
        leaf_kwargs: The kwargs for the leaves.

    """
    # Import the locally to circumvent circular imports.
    from simple_einet.layers.distributions.binomial import Binomial
    from simple_einet.layers.distributions.categorical import Categorical
    from simple_einet.layers.distributions.multivariate_normal import MultivariateNormal
    from simple_einet.layers.distributions.normal import Normal, RatNormal
    from simple_einet.layers.distributions.piecewise_linear import PiecewiseLinear

    if dist == Dist.NORMAL:
        leaf_type = Normal
        leaf_kwargs = {}
    elif dist == Dist.NORMAL_RAT:
        leaf_type = RatNormal
        leaf_kwargs = {"min_sigma": cfg.min_sigma, "max_sigma": cfg.max_sigma}
    elif dist == Dist.BINOMIAL:
        leaf_type = Binomial
        leaf_kwargs = {"total_count": 2**cfg.n_bits - 1}
    elif dist == Dist.CATEGORICAL:
        leaf_type = Categorical
        leaf_kwargs = {"num_bins": 2**cfg.n_bits - 1}
    elif dist == Dist.MULTIVARIATE_NORMAL:
        leaf_type = MultivariateNormal
        leaf_kwargs = {"cardinality": cfg.multivariate_cardinality}
    elif dist == Dist.PIECEWISE_LINEAR:
        leaf_type = PiecewiseLinear
        leaf_kwargs = {}
    else:
        raise ValueError(f"Unknown distribution ({dist}).")
    return leaf_kwargs, leaf_type
