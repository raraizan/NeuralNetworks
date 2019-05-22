import numpy

def get_category(a):
    return numpy.argmax(a)

def categorize(j):
    e = numpy.zeros((10,1))
    e[j] = 1.0
    return e
