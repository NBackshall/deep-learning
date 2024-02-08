"""Sequence of transformations."""
import numpy as np
from typing import List

from .transformation import Transformation


class Sequence:
    """Sequence of transformations."""

    def __init__(self, layers: List[Transformation]):
        """Init."""
        self._layers = layers

    def forward(self, x):
        """Forward pass."""
        for layer in self._layers:
            x = layer.forward(x)
        return x
