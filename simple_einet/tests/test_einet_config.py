#!/usr/bin/env python3

import unittest
from simple_einet.distributions import Normal

from simple_einet.einet import EinetConfig
from simple_einet.type_checks import OutOfBoundsException


class TestEinetConfig(unittest.TestCase):
    def setUp(self) -> None:
        self.config = EinetConfig(
            in_shape=(8, 8),
            num_sums=5,
            num_leaves=5,
            num_repetitions=5,
            num_classes=5,
            depth=5,
            dropout=0.5,
            leaf_type=Normal,
            leaf_kwargs={},
        )
        self.config.assert_valid()
        return super().setUp()

    def test_illegal_in_shape(self):
        self.config.in_shape = [-1, 1]

        with self.assertRaises(OutOfBoundsException):
            self.config.assert_valid()

        self.config.in_shape = [1, -1]

        with self.assertRaises(OutOfBoundsException):
            self.config.assert_valid()

    def test_illegal_depth(self):
        self.config.depth = -1

        with self.assertRaises(OutOfBoundsException):
            self.config.assert_valid()

    def test_illegal_num_leaves(self):
        self.config.num_leaves = -1

        with self.assertRaises(OutOfBoundsException):
            self.config.assert_valid()

    def test_illegal_dropout(self):
        self.config.dropout = -0.5

        with self.assertRaises(OutOfBoundsException):
            self.config.assert_valid()

    def test_illegal_num_classes(self):
        self.config.num_classes = -1

        with self.assertRaises(OutOfBoundsException):
            self.config.assert_valid()

    def test_illegal_num_repetitions(self):
        self.config.num_repetitions = -1

        with self.assertRaises(OutOfBoundsException):
            self.config.assert_valid()

    def test_illegal_num_sums(self):
        self.config.num_sums = -1

        with self.assertRaises(OutOfBoundsException):
            self.config.assert_valid()

    def test_illegal_depth_inshape_combination(self):
        self.config.depth = 10
        self.config.in_shape = (2, 2)

        with self.assertRaises(Exception):
            self.config.assert_valid()


if __name__ == "__main__":
    unittest.main()
