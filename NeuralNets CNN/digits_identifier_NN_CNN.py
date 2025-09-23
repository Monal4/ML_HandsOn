import numpy as np
import random
import keras


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


def main():
    network = Network([784, 70, 10])

    training_dataset = keras.datasets.mnist.load_data()[0]
    testing_dataset = keras.datasets.mnist.load_data()[1]

    training_dataset_zip = convertLabelsAndZipData(training_dataset, True)
    testing_dataset_zipped = convertLabelsAndZipData(testing_dataset, False)

    network.model(training_dataset_zip, 100, 10, 0.3, testing_dataset_zipped)


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
