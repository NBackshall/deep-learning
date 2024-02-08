"""Example creating nn."""
import numpy as np

from implementations import Linear
from implementations import Sequence
from implementations import NeuralNetwork


# Set print options to 2 d.p.
np.set_printoptions(precision=2)

in_features, out_features = 10, 5
x = np.random.rand(in_features)
linear = Linear(in_features, out_features)
y = linear.forward(x)

print("One layer:")
print(f"Input: {x}")
print(f"Output: {y}")

layers = Sequence([
    Linear(in_features, 10),
    Linear(10, 100),
    Linear(100, 100),
    Linear(100, out_features),
])

nn = NeuralNetwork(layers)
y = nn.forward(x)

print("Deep nerual network:")
print(f"Input: {x}")
print(f"Output: {y}")
