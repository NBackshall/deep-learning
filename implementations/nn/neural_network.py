"""Neural network."""
import numpy as np

from .sequence import Sequence


class NeuralNetwork:
    """Nerual Network."""

    def __init__(self, layers: Sequence):
        """Init."""
        self._layers = layers

    def forward(self, x):
        """Forward pass."""
        return self._layers.forward(x)
