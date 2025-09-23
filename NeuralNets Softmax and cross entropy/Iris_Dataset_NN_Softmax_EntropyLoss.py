import numpy as np
import random
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler


def ReLU(z):
    return np.maximum(0, z)


def ReLU_Prime(z):
    return (z > 0).astype(float)


def softmax(z):
    exp = np.exp(z - np.max(z))
    return exp / np.sum(exp)


class Network:

    def __init__(self, layers):
        self.layers = layers
        self.numberOfLayers = len(layers)
        self.weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]
        self.biases = [np.random.randn(y, 1) for y in layers[1:]]

    def model(self, training_set, epoch, batch_size, learning_rate, testing_set):
        for i in range(epoch):
            random.shuffle(training_set)
            mini_batches = [
                training_set[k: k + batch_size] for k in range(0, len(training_set))
            ]
            for batch in mini_batches:
                self.update_weights_biases(batch, learning_rate)
            print("Epoch {0}: {1} / {2}".format(
                i, self.evaluate(testing_set), len(testing_set)))

    def update_weights_biases(self, batch, learning_rate):
        delta_weights = [np.zeros(weight.shape) for weight in self.weights]
        delta_biases = [np.zeros(bias.shape) for bias in self.biases]

        for x, y in batch:
            derived_weights_from_backprop, derived_biases_from_backprop = self.backprop(x, y)
            delta_weights = [w + dw for w, dw in zip(delta_weights, derived_weights_from_backprop)]
            delta_biases = [b + db for b, db in zip(delta_biases, derived_biases_from_backprop)]
        self.weights = [w - (learning_rate / len(batch)) * dw for w, dw in zip(self.weights, delta_weights)]
        self.biases = [b - (learning_rate / len(batch)) * db for b, db in zip(self.biases, delta_biases)]

    def backprop(self, x, y):
        derived_weights = [np.zeros(weight.shape) for weight in self.weights]
        derived_biases = [np.zeros(bias.shape) for bias in self.biases]

        zs = []

        activation = x.reshape(4, 1)
        activations = [activation]

        # ReLU - mid layers
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            z = np.dot(weight, activation) + bias
            zs.append(z)
            activation = ReLU(z)
            activations.append(activation)

        # softmax - last layer
        z = np.dot(self.weights[-1], activation) + self.biases[-1]
        zs.append(z)
        activation = softmax(z)
        activations.append(activation)

        derivative = activations[-1] - y
        derived_biases[-1] = derivative
        derived_weights[-1] = np.dot(derivative, activations[-2].transpose())

        for i in range(2, self.numberOfLayers):
            derivative = np.dot(self.weights[-i + 1].transpose(), derivative) * ReLU_Prime(zs[-i])
            derived_biases[-i] = derivative
            derived_weights[-i] = np.dot(derivative, activations[-i - 1].transpose())

        return derived_weights, derived_biases

    def evaluate(self, testing_set):
        result = [(np.argmax(self.feed_forward(x)), y) for x, y in testing_set]
        return sum(int(x == y) for (x, y) in result)

    def feed_forward(self, x):
        for i in range(len(self.weights) - 1):
            z = np.dot(self.weights[i], x) + self.biases[i]
            x = ReLU(z)

        z = np.dot(self.weights[-1], x) + self.biases[-1]
        x = softmax(z)

        return x


def convertLabels(dataset):
    labels = dataset.target
    scaler = MinMaxScaler()
    data = scaler.fit_transform(dataset.data)
    converted_labels = np.zeros((len(labels), 3))
    for i in range(len(labels)):
        label = labels[i]
        converted_labels[i][label] = 1

    converted_labels = [label.reshape(3, 1) for label in converted_labels]

    return list(zip(data, converted_labels))


def main():
    network = Network([4, 17, 13, 3])

    dataset = datasets.load_iris()
    scaler = MinMaxScaler()
    data = scaler.fit_transform(dataset.data)

    training_set = convertLabels(dataset)
    testing_set = list(zip(data, dataset.target))

    network.model(training_set, 40, 15, 2, testing_set)


if __name__ == "__main__":
    main()
