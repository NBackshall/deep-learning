"""Example using perceptron."""
import numpy as np
import matplotlib.pyplot as plt

from implementations import (
    Perceptron,
    l2_loss,
)


# Set print options to 2 d.p.
np.set_printoptions(precision=2)

def sample(dataset, batch_size):
    indexes = np.array(range(len(dataset)))
    np.random.shuffle(indexes)
    return dataset[indexes[:batch_size]]

def show_plot(perceptron, dataset):
    plt.clf()
    x, y = zip(*dataset)
    plt.scatter(x, y, label='Data', color='blue')
    y = perceptron.weights * x + perceptron.bias
    plt.plot(x, y, label='Perceptron', color='red')

    plt.title('Learning linear data with MSE loss')
    plt.legend()
    plt.xlim(*xlim)
    plt.ylim(*ylim)

    plt.show(block=False)
    plt.pause(1e-4)

dataset_size = 100
batch_size = dataset_size // 5

x = np.linspace(-5.0, 5.0)
y = (x * np.random.uniform(-1, 1) * np.random.uniform(-1, 1)) + np.random.sample(x.shape)

m = max(abs(min(y)), abs(max(y))) * 1.5
xlim = (-5, 5)
ylim = (-m, +m)

dataset = np.array(list(zip(x, y)))

perceptron = Perceptron(1)

training_steps = 200
lr = 0.07

for _ in range(training_steps):
    batch = sample(dataset, batch_size)
    x, y = zip(*batch)
    x = np.array([x]).T
    predictions = perceptron.forward(x)
    loss = np.mean(
        l2_loss(predictions, y)
    )
    dL_dp = 2*(predictions - y)
    dL_dw = x*dL_dp
    dL_db = dL_dp
    perceptron.weights += - lr * np.mean(dL_dw)
    perceptron.bias += - lr * np.mean(dL_db)
    show_plot(perceptron, dataset)
