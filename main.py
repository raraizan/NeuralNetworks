import matplotlib.pyplot as plt
import numpy
import random
import json

from pprint import pprint

from nn import MultiLayeredPerceptron, NeuralNetwork
from nn.data_loader import load_data, load_data_wrapper
from nn.misc import *

DATASETS_PATH = 'flask_app/datasets/mnist.pkl.gz'

# # Load tagged data into memory
training_data, validation_data, test_data = load_data_wrapper(DATASETS_PATH)

first_nn = NeuralNetwork().create((784, 100, 40, 10), activation_function='sigmoid')


# Pick random 
# sample = random.choice(training_data)
# print(sample[0])
# result = first_nn.get_category(sample[0])
# print(result)
first_nn.train_model(training_data, 30, 25, 0.1, test_data=test_data)
first_nn.save()