import mnist_loader
import network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = network.Network([784, 100, 10])
net.SGD(training_data=training_data, epochs=30, mini_batch_size=10, eta=0.01, test_data=test_data)
