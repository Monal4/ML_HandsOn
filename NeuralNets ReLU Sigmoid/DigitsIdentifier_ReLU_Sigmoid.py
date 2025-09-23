import random

import numpy as np
import keras


def Relu(z):
    return np.maximum(0, z)


def Relu_derivative(z):
    return (z > 0).astype(float)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))


class Network:
    def __init__(self, layers):
        self.numOfLayers = len(layers)
        self.weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]
        self.biases = [np.random.randn(y, 1) for y in layers[1:]]

    def stochastic_gradient_descend(self, training_set, epoch, batch_size, learning_rate, test_set):
        for i in range(epoch):
            random.shuffle(training_set)
            mini_batches = [
                training_set[k: k + batch_size]
                for k in range(0, len(training_set), batch_size)
            ]
            for batch in mini_batches:
                self.update_weights_and_biases(batch, learning_rate)
            print("Epoch {0}: {1} / {2}".format(
                i, self.evaluate(test_set), len(test_set)))

    def evaluate(self, test_data):
        result = [(np.argmax(self.feed_forward(x)), y) for x, y in test_data]
        return sum(int(x == y) for (x, y) in result)

    def update_weights_and_biases(self, batch, learning_rate):
        batch_weights = [np.zeros(weight.shape) for weight in self.weights]
        batch_biases = [np.zeros(bias.shape) for bias in self.biases]

        for x, y in batch:
            derivative_weights, derivative_biases = self.backprop_relu(x, y)
            batch_weights = [w + dw for w, dw in zip(batch_weights, derivative_weights)]
            batch_biases = [b + db for b, db in zip(batch_biases, derivative_biases)]

        self.weights = [w - ((dw / len(batch)) * learning_rate) for w, dw in zip(self.weights, batch_weights)]
        self.biases = [b - ((db / len(batch)) * learning_rate) for b, db in zip(self.biases, batch_biases)]

    def backprop_relu(self, x, y):
        x_weights = [np.zeros(weight.shape) for weight in self.weights]
        x_biases = [np.zeros(bias.shape) for bias in self.biases]

        activation = x
        activations = [activation]
        zs = []

        for i in range(len(self.weights) - 1):
            z = np.dot(self.weights[i], activation) + self.biases[i]
            zs.append(z)
            activation = Relu(z)
            activations.append(activation)

        z = np.dot(self.weights[-1], activation) + self.biases[-1]
        zs.append(z)
        activation = sigmoid(z)
        activations.append(activation)

        derivative = 2 * (activations[-1] - y) * sigmoid_derivative(zs[-1])

        x_biases[-1] = derivative
        x_weights[-1] = np.dot(derivative, activations[-2].transpose())

        for l in range(2, self.numOfLayers):
            derivative = np.dot(self.weights[-l + 1].transpose(), derivative) * Relu_derivative(zs[-l])
            x_weights[-l] = np.dot(derivative, activations[-l - 1].transpose())
            x_biases[-l] = derivative

        return x_weights, x_biases

    def feed_forward(self, x):

        for i in range(len(self.weights) - 1):
            z = np.dot(self.weights[i], x) + self.biases[i]
            x = Relu(z)

        z = np.dot(self.weights[-1], x) + self.biases[-1]
        x = sigmoid(z)

        return x


def main():
    # NOTE P0 Starts
    net = Network([784, 70, 10])

    training_dataset = keras.datasets.mnist.load_data()[0]
    testing_dataset = keras.datasets.mnist.load_data()[1]

    training_dataset_zip = convertLabelsAndZipData(training_dataset, True)
    testing_dataset_zipped = convertLabelsAndZipData(testing_dataset, False)

    net.stochastic_gradient_descend(training_dataset_zip, 30, 10, 3, testing_dataset_zipped)


def convertLabelsAndZipData(dataset, training):
    features = dataset[0]
    labels = dataset[1]
    if training:
        converted_labels = np.zeros((len(labels), 10))

        for i in range(len(converted_labels)):
            converted_labels[i][labels[i]] = 1

        labels = [label.reshape(10, 1) for label in converted_labels]

    features = [feature.reshape((28 * 28, 1)) for feature in features]
    features = [feature / 255 for feature in features]

    return list(zip(features, labels))


if __name__ == "__main__":
    main()
