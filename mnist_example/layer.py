import numpy as np
import matplotlib.pyplot as plt
import function


def init_network():
    network = {}
    network['W1'] = np.random.randn(784)
    network['B1'] = np.zeros(256)
    network['W2'] = np.random.randn(256)
    network['B2'] = np.zeros(256)
    network['W3'] = np.random.randn(256)
    network['B3'] = np.zeros(10)

    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['B1'], network['B2'], network['B3']

    a1 = np.dot(x, W1) + b1
    z1 = relu.forward(a1)

    a2 = np.dot(z1, W2) + b2
    z2 = relu.forward(a2)

    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

