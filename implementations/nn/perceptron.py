"""Simple Perceptron implementation in Python."""
import numpy as np

class Perceptron:
    """Simple Perceptron implementation in Python."""

    def __init__(self, input_size: int, learning_rate: float = 0.01):
        """Initialise the Perceptron.

        Args:
            input_size (int): The number of input features.
            learning_rate (float, optional): The learning rate. Defaults to 0.01.
        """
        self.weights = np.random.rand(input_size)
        self.bias = np.random.random()
        self.learning_rate = learning_rate

    def forward(self, x):
        """Forward pass of perceptron

        Args:
            x (float): Input into perceptron.
        """
        assert (
            len(x) == len(self.weights)
        ), "Length of input must equal length of weights."
        y = self.bias + np.matmul(x, self.weights)
        return y
