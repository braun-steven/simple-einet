from unittest import TestCase

import torch
from parameterized import parameterized

from simple_einet.abstract_layers import logits_to_log_weights
from simple_einet.layers.linsum import LinsumLayer
from simple_einet.tests.layers.test_utils import get_sampling_context


class TestLinsumLayer(TestCase):
    def setUp(self) -> None:
        self.layer = LinsumLayer(num_features=4, num_sums_in=3, num_sums_out=2, num_repetitions=5)

    def test_logits_to_log_weights(self):
        for dim in range(self.layer.logits.dim()):
            log_weights = logits_to_log_weights(self.layer.logits, dim=dim)
            sums = log_weights.logsumexp(dim=dim)
            target = torch.zeros_like(sums)
            self.assertTrue(torch.allclose(sums, target, atol=1e-5))

    def test_forward_shape(self):
        bs = 2
        x = torch.randn(bs, self.layer.num_features, self.layer.num_sums_in, self.layer.num_repetitions)
        out = self.layer(x)
        self.assertEqual(
            out.shape, (bs, self.layer.num_features_out, self.layer.num_sums_out, self.layer.num_repetitions)
        )

    @parameterized.expand([(False,), (True,)])
    def test__condition_weights_on_evidence(self, differentiable: bool):
        bs = 2
        x = torch.randn(bs, self.layer.num_features, self.layer.num_sums_in, self.layer.num_repetitions)
        self.layer._enable_input_cache()
        self.layer(x)

        ctx = get_sampling_context(layer=self.layer, num_samples=bs, is_differentiable=differentiable)
        log_weights = self.layer._select_weights(ctx, self.layer.logits)
        log_weights = self.layer._condition_weights_on_evidence(ctx, log_weights)
        sums = log_weights.logsumexp(dim=2)
        target = torch.zeros_like(sums)
        self.assertTrue(torch.allclose(sums, target, atol=1e-5))

    @parameterized.expand([(False,), (True,)])
    def test__sample_from_weights(self, differentiable: bool):
        N = 2
        ctx = get_sampling_context(layer=self.layer, num_samples=N, is_differentiable=differentiable)
        log_weights = self.layer._select_weights(ctx, self.layer.logits)
        indices = self.layer._sample_from_weights(ctx, log_weights)
        if differentiable:
            self.assertEqual(tuple(indices.shape), (N, self.layer.num_features, self.layer.num_sums_in))
        else:
            self.assertEqual(tuple(indices.shape), (N, self.layer.num_features))

    @parameterized.expand([(False,), (True,)])
    def test__select_weights(self, differentiable: bool):
        N = 2
        ctx = get_sampling_context(layer=self.layer, num_samples=N, is_differentiable=differentiable)
        weights = self.layer._select_weights(ctx, self.layer.logits)
        self.assertEqual(tuple(weights.shape), (N, self.layer.num_features_out, self.layer.num_sums_in))
