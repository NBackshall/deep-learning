"""Linear layer."""
import numpy as np

from .transformation import Transformation


class Linear(Transformation):
    """Linear layer."""

    def __init__(self, in_features, out_features):
        """Initialize weights and biases."""
        super().__init__(in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features
        self.weights = np.random.randn(in_features, out_features)
        self.biases = np.random.randn(in_features)

    def forward(self, x):
        """Forward pass."""
        assert (
            len(x) == self.in_features
        ), "Length of input must equal length of weights."
        y =  np.matmul(x + self.biases, self.weights)
        return y
