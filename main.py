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
first_nn.create((784, 40, 10), activation_function=sigmoid)


# Pick random 
sample = random.choice(training_data)
print(first_nn.feedforward(sample[0]))

first_nn.train_model(training_data, 30, 30, 0.1)

print(first_nn.feedforward(sample[0]))
print(sample[1])
image = numpy.reshape(sample[0][:,0], (28, 28))
plt.imshow(image)
plt.show()