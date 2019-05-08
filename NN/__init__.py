import random
import json
import datetime

import numpy as np

from .activation_functions import heaviside, sigmoid
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
        self.activation_function = heaviside

    def create(self, sizes):
        self.num_layers = len(sizes) 
        self.sizes = tuple(sizes)
        self.biases = np.array([np.random.randn(y, 1) for y in sizes[1:]])
        self.weights = np.array(
            [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])],
        )
        self.initialized = True
        return self
    
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
            self.weights = [np.array(w) for w in data["weights"]] 
            self.biases = [np.array(b) for b in data["biases"]]
        self.initialized = True
        return self
    
    def feedforward(self, a):
        if not self.initialized:
            raise PerceptronNotInitialized
        
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a



class NeuralNetwork(MultiLayeredPerceptron):
    pass

    # def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
    #     """
    #     Se entrena la red neuronal usando descenso gradiente estocastico
    #     en un mini-batch. ``training_data`` es una lista de tuplas
    #     ``(x, y)`` donde ``x`` es la entrada de datos y ``y`` es el
    #     resultado deseado. Si se provee ``test_data`` se evaluara el
    #     desempeño de la red al final de cada iteracion. Esto ayuda a
    #     monitorear el progreso del entrenamiento.
    #     """

    #     training_data = list(training_data)
    #     n = len(training_data)

    #     if test_data:
    #         test_data = list(test_data)
    #         n_test = len(test_data)

    #     for j in range(epochs):
    #         random.shuffle(training_data)
    #         mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]

    #         for mini_batch in mini_batches:
    #             self.update_parameters(mini_batch, eta)

    #         if test_data:
    #             percentil = 100 * (self.evaluate(test_data) / n_test)
    #             self.history.update( {j: [percentil, self.weights]})
    #             print("Epoch {} : {} accurate".format(j, percentil))
    #         else:
    #             print("Epoch {} complete".format(j))

    # def update_parameters(self, mini_batch, eta):
    #     """
    #     Actualiza ``weights`` y ``biases`` aplicando aplicando
    #     descenso gradiente calculado mediante el algoritmo de
    #     backpropagation, aplicado a un solo mini-batch compuesto
    #     por tuplas ``(x, y)`` con ``x`` los datos de entrada y 
    #     ``y`` el resultado deseado. Se utiliza ``eta`` como
    #     ritmo de aprendizaje.
    #     """

    #     nabla_b = [np.zeros(b.shape) for b in self.biases]
    #     nabla_w = [np.zeros(w.shape) for w in self.weights]

    #     for x, y in mini_batch:
    #         delta_nabla_b, delta_nabla_w = self.backprop(x, y)

    #         nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
    #         nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

    #     self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
    #     self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    # def backprop(self, x, y):
    #     """
    #     Regresa una tupla ``(nabla_b, nabla_w)`` que representa
    #     el gradiente de la funcion de costo C_x. ``nabla_b`` y
    #     ``nabla_w`` son, capa por capa, listas de arreglos,
    #     similares a ``self.biases`` y ``self.weights``.
    #     """

    #     nabla_b = [np.zeros(b.shape) for b in self.biases]
    #     nabla_w = [np.zeros(w.shape) for w in self.weights]

    #     # feedforward
    #     activation = x
    #     activations = [x] # Lista para gardar todas las activaciones capa por capa
    #     zs = [] # lista para guardar los vectores z capa por capa

    #     for b, w in zip(self.biases, self.weights):
    #         z = np.dot(w, activation) + b
    #         zs.append(z)
    #         activation = sigmoid(z)
    #         activations.append(activation)

    #     # Paso hacia atras
    #     delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
    #     nabla_b[-1] = delta
    #     nabla_w[-1] = np.dot(delta, activations[-2].transpose())


    #     for l in range(2, self.num_layers):
    #         z = zs[-l]
    #         sp = sigmoid_prime(z)
    #         delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
    #         nabla_b[-l] = delta
    #         nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

    #     return (nabla_b, nabla_w)

    # def evaluate(self, test_data):
    #     """
    #     Regresa el numero de entradas para las cuales la red entrego
    #     el resultado correcto. Nota que la salida de la red neuronal
    #     es el indice de la neurona con mayor activacion en la capa
    #     final.
    #     """
    #     test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]

    #     return sum(int(x == y) for (x, y) in test_results)

    # def cost_derivative(self, output_activations, y):
    #     """
    #     Regresa el vector de derivadas parciales para las
    #     activaciones de salida.
    #     """
    #     return (output_activations-y)