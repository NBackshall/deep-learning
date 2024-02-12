"""Sigmoid activation function."""
import numpy as np

from ..nn.transformation import Transformation


class Sigmoid(Transformation):
    """Linear layer."""

    def __init__(self, in_features):
        """Initialize weights and biases."""
        super().__init__(in_features, in_features)
        self.in_features = in_features

    def forward(self, x):
        """Forward pass."""
        assert (
            len(x) == self.in_features
        ), "Length of input must equal length of weights."
        return 1 / (1 + np.exp(-x))
