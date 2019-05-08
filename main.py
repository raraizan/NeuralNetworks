import matplotlib.pyplot as plt
import numpy
import random

from pprint import pprint

from nn import MultiLayeredPerceptron, NeuralNetwork
from nn.data_loader import load_data, load_data_wrapper
from nn.activation_functions import sigmoid, heaviside
# from nn import 

DATASETS_PATH = 'datasets/mnist.pkl.gz'

# Load tagged data into memory
training_data, validation_data, test_data = load_data_wrapper(DATASETS_PATH)

first_nn = NeuralNetwork()
first_nn.create((784, 10), activation_function=sigmoid)

# images = tuple(numpy.reshape(flat_data, (28, 28)) for flat_data in test_data[0])

# Pick random 
sample = random.choice(training_data)

print(first_nn.feedforward(sample[0]))