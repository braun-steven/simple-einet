import torch
from torch.distributions.utils import probs_to_logits
from torch import distributions as dist
from torch import nn
from torch.nn import functional as F

from simple_einet.layers.distributions.abstract_leaf import AbstractLeaf
from simple_einet.sampling_utils import SamplingContext


class Categorical(AbstractLeaf):
    """Categorical layer. Maps each input feature to its categorical log likelihood.

    Probabilities are modeled as unconstrained parameters and are transformed via a softmax function into [0, 1] when needed.
    """

    def __init__(self, num_features: int, num_channels: int, num_leaves: int, num_repetitions: int, num_bins: int):
        """
        Initializes a categorical distribution with the given parameters.

        Args:
            num_features (int): The number of features in the input data.
            num_channels (int): The number of channels in the input data.
            num_leaves (int): The number of leaves in the tree structure.
            num_repetitions (int): The number of repetitions for each leaf.
            num_bins (int): The number of bins for the categorical distribution.
        """
        super().__init__(num_features, num_channels, num_leaves, num_repetitions)

        # Create logits
        p = 0.5 + (torch.rand(1, num_channels, num_features, num_leaves, num_repetitions, num_bins) - 0.5) * 0.2
        self.logits = nn.Parameter(probs_to_logits(p))

    def _get_base_distribution(self, ctx: SamplingContext = None):
        # Use sigmoid to ensure, that probs are in valid range
        return dist.Categorical(logits=F.log_softmax(self.logits, dim=-1))
