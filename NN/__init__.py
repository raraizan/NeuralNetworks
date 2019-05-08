import random
import json
import datetime

import numpy

from .activation_functions import heaviside, sigmoid, sigmoid_prime
from .exceptions import PerceptronNotInitialized


class Perceptron:
    def __init__(self, weights, bias, activation_function=None):
        self.weights = weights
        self.bias = bias
        self.activation_function = activation_function if activation_function else heaviside

    def feedforward(self, input_vector):
        value = self.activation_function(self.weights.dot(input_vector))
        return value


class MultiLayeredPerceptron:
    def __init__(self):
        self.initialized = False

    def create(self, sizes, activation_function=None):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [numpy.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [numpy.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.activation_function = activation_function if activation_function else heaviside

        self.initialized = True
        return self

    def feedforward(self, a):
        if not self.initialized:
            raise PerceptronNotInitialized
        
        for b, w in zip(self.biases, self.weights):
            a = self.activation_function(numpy.dot(w, a) + b)
        return a

    def save(self, path=None, prefix=None):
        if not self.initialized:
            raise PerceptronNotInitialized
        
        data = {
            "sizes": self.sizes,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
        }
        
        prefix = "{}_".format(prefix) if prefix else ""
        path = '{}/'.format(path) if path else ''
        time_stamp = datetime.datetime.now().strftime('%m_%d_%Y_%H:%M:%S')

        filename = "{}{}model_{}.json".format(path, prefix, time_stamp)

        f = open(filename, 'w+')
        json.dump(data, f)
        f.close() 

    def load(self, filename): 
        with open(filename, "r") as f: 
            data = json.load(f) 
            f.close() 
 
            sizes = data["sizes"] 
 
            self.num_layers = len(sizes) 
            self.sizes = tuple(sizes)
            self.weights = [numpy.array(w) for w in data["weights"]] 
            self.biases = [numpy.array(b) for b in data["biases"]]
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

            # if test_data:
            #     percentil = 100 * (self.evaluate(test_data) / n_test)
            #     self.history.update( {j: [percentil, self.weights]})
                # print("Epoch {} : {} accurate".format(j, percentil))
            # else:
            #     print("Epoch {} complete".format(j))
            print("Epoch {} complete".format(j))

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
        """
        Regresa una tupla ``(nabla_b, nabla_w)`` que representa
        el gradiente de la funcion de costo C_x. ``nabla_b`` y
        ``nabla_w`` son, capa por capa, listas de arreglos,
        similares a ``self.biases`` y ``self.weights``.
        """

        nabla_b = [numpy.zeros(b.shape) for b in self.biases]
        nabla_w = [numpy.zeros(w.shape) for w in self.weights]

        # feedforward
        activation = sample
        activations = [sample] # Lista para gardar todas las activaciones capa por capa
        zs = [] # lista para guardar los vectores z capa por capa

        for b, w in zip(self.biases, self.weights):
            z = numpy.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # Paso hacia atras
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
        """
        Regresa el numero de entradas para las cuales la red entrego
        el resultado correcto. Nota que la salida de la red neuronal
        es el indice de la neurona con mayor activacion en la capa
        final.
        """
        test_results = [(numpy.argmax(self.feedforward(x)), y) for (x, y) in test_data]

        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """
        Regresa el vector de derivadas parciales para las
        activaciones de salida.
        """
        return output_activations - y