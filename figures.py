import numpy

from tryhardnn import Perceptron


weights = numpy.array([1, 2, 3, 4, 5])

a = Perceptron(weights, 1)

x = numpy.zeros_like(weights)
print(a.feedforward(x))
