import numpy as np

# Functions here must be compatible with numpy's ndarray


def sigmoid(x):
    """
    Sigmoid function
    """
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_prime(x):
    """
    Sigmoid's derivative
    """
    return sigmoid(x) * (1.0 - sigmoid(x))


def heaviside(x):
    """
    Heaviside function
    """
    return 0.5 * (np.sign(x) + 1.0)
