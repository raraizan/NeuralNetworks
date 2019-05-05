import NN.neural_network
import NN.mnist_loader
import datetime

net = neural_network.MLP([784, 100, 30, 10])

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net.SGD(training_data, 100, 25, 0.1, test_data=validation_data)

net.save(
    'models/MLP_{}.json'.format(
        datetime.datetime.now(),
    ),
)
