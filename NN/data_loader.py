import pickle
import gzip

import numpy


def load_data(path):
    with gzip.open(path, 'rb') as f: 
        training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
        return (training_data, validation_data, test_data)

def load_data_wrapper(path):
    tr_d, va_d, te_d = load_data(path)

    training_inputs = [numpy.reshape(x, (784, 1)) for x in tr_d[0]]
    training_tag = [categorize(x) for x in tr_d[1]]
    training_data = tuple(zip(training_inputs, training_tag))

    validation_inputs = [numpy.reshape(x, (784, 1)) for x in va_d[0]]
    validation_tags = [categorize(x) for x in va_d[1]]

    validation_data = tuple(zip(validation_inputs, validation_tags))

    test_inputs = [numpy.reshape(x, (784, 1)) for x in te_d[0]]
    test_tags = [categorize(x) for x in te_d[1]]

    test_data = tuple(zip(test_inputs, test_tags))

    return training_data, validation_data, test_data

def categorize(j):
    e = numpy.zeros((10, 1))
    e[j] = 1.0
    return e
