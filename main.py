import numpy
from pprint import pprint
import nn

mlp = nn.MultiLayeredPerceptron()
mlp.create([2,4,1])
# mlp.load('models/model_05_07_2019_19:01:46.json')

vec = numpy.array([1,4])

result = mlp.feedforward(vec)
pprint(result)