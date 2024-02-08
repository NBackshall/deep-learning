"""Base class for transformation."""
from abc import ABC, abstractmethod


class Transformation:
    """Base class for transformation."""

    def __init__(self, in_features, out_features):
        """Init."""
        self._in_features = in_features
        self._out_features = out_features

    @abstractmethod
    def forward(self, x):
        """Forward pass."""
        pass
