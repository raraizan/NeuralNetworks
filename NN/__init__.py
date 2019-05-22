import random
import json
import datetime

import numpy

from .activation_functions import heaviside, sigmoid, sigmoid_prime
from .cost_funtions import square_error
from .exceptions import InitializationError
from .misc import *

ACTIVATION_FUNCTIONS = {
    'sigmoid': sigmoid,
    'heaviside': heaviside,
}

# Perceptron class
class Perceptron:
    def __init__(self, weights, bias, activation_function=None):
        self.weights = numpy.array(weights)
        self.bias = bias
        self.activation_function = ACTIVATION_FUNCTIONS[activation_function] if activation_function else heaviside
        
    def __str__(self):
        print("Perceptron initialized with:")
        print("{} weights, {}".format(len(self.weights), self.weights))
        print("{} as bias".format(self.bias))

    def feedforward(self, input_vector):
        value = self.activation_function(self.weights.dot(input_vector) - self.bias)
        return value
    
    def train(self, dataset, subsets=10, max_iters=100, eps=1e-16, cost_function=None):
        
        cost_funtion = cost_function if cost_function else square_error

        counter = 0
        epsilon = 1e-16

        self.error_history = []
        self.params_history = []
        best_params = self.weights, self.bias
        best_error = cost_funtion(dataset, self.feedforward)
        print("initial error: {}".format(best_error))

        while best_error > eps and counter < max_iters:
            # Generate a set of variations from the initial params
            param_variations = [
                (
                    numpy.array(
                        tuple([random.gauss(weight, best_error) for weight in best_params[0]]),
                    ),
                    random.gauss(best_params[1], best_error),
                ) for _ in range(10)]

            # Evaluate all and find the best
            for i, params in enumerate(param_variations):
                self.weights, self.bias = params

                current_error = cost_funtion(dataset, self.feedforward)
                if  current_error <= best_error:
                    best_error = current_error
                    best_params = params
            
            self.error_history.append(best_error)
            self.params_history.append(best_params)

            counter += 1
        
        final_error = square_error(dataset, self.feedforward)
        print("final error: {} reached in {} iterations".format(final_error, counter))

        return self.error_history, self.params_history


class MultiLayeredPerceptron:
    def __init__(self):
        self.initialized = False

    def create(self, sizes, activation_function=None):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [numpy.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [numpy.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

        self.activation_function_key = activation_function if activation_function else 'heaviside'
        self.activation_function = ACTIVATION_FUNCTIONS[self.activation_function_key]

        self.initialized = True
        return self

    def feedforward(self, input_vector):
        if not self.initialized:
            raise InitializationError

        for bias, weights in zip(self.biases, self.weights):
            input_vector = self.activation_function(numpy.dot(weights, input_vector) + bias)
        return input_vector

    def save(self, path=None, prefix=None):
        if not self.initialized:
            raise InitializationError
        
        data = {
            "sizes": self.sizes,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
            "activation_function": self.activation_function_key,
        }

        time_stamp = datetime.datetime.now().strftime('%m_%d_%Y_%H:%M:%S')
        prefix = "{}_{}_{}".format(prefix, self.activation_function_key, time_stamp) if prefix else "{}_{}".format(self.activation_function_key, time_stamp)
        path = '{}/{}'.format(path, prefix) if path else prefix
        filename = "{}.json".format(path)

        with open(filename, 'w+') as f:
            json.dump(data, f)

    def load(self, filename, path=None):
        path = '{}/{}'.format(path, filename) if path else filename
        with open(path, "r") as f: 
            data = json.load(f)
 
            sizes = data["sizes"] 
 
            self.num_layers = len(sizes) 
            self.sizes = tuple(sizes)
            self.weights = [numpy.array(w) for w in data["weights"]] 
            self.biases = [numpy.array(b) for b in data["biases"]]
            self.activation_function_key = data["activation_function"]
            self.activation_function = ACTIVATION_FUNCTIONS[self.activation_function_key]

        self.initialized = True
        return self


class NeuralNetwork(MultiLayeredPerceptron):
    def train_model(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_parameters(mini_batch, eta)

            if test_data:
                percentil = (100.0 / n_test) * self.evaluate(test_data)
                print("Epoch {} : {} accurate".format(j, percentil))
            else:
                print("Epoch {} complete".format(j + 1))

    def update_parameters(self, mini_batch, eta):
        nabla_b = [numpy.zeros_like(b) for b in self.biases]
        nabla_w = [numpy.zeros_like(w) for w in self.weights]

        for sample, tag in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(sample, tag)

            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, sample, tag):
        nabla_b = [numpy.zeros(b.shape) for b in self.biases]
        nabla_w = [numpy.zeros(w.shape) for w in self.weights]

        # feedforward
        activation = sample
        activations = [sample]
        zs = []

        for b, w in zip(self.biases, self.weights):
            z = numpy.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # Back step
        delta = self.cost_derivative(activations[-1], tag) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = numpy.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = numpy.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = numpy.dot(delta, activations[-l-1].transpose())

        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(get_category(self.feedforward(x)), get_category(y)) for (x, y) in test_data]

        value = 0

        for i, j in test_results:
            if i == j:
                value += 1

        return value

    def cost_derivative(self, output_activations, y):
        return output_activations - y
