"""Example using perceptron."""
import numpy as np

from implementations import Perceptron


# Set print options to 2 d.p.
np.set_printoptions(precision=2)

input_size = 10
x = np.random.rand(input_size)
perceptron = Perceptron(input_size)
y = perceptron.forward(x)

print(f"Input: {x}")
print(f"Output: {y}")
print(f"Perceptron weights: {perceptron.weights}")
print(f"Perceptron bias: {perceptron.bias}")
