import random

import numpy as np
import keras


class Network(object):

    def __init__(self, layers):
        self.weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]
        self.biases = [np.random.randn(y, 1) for y in layers[1:]]
        self.number_of_layers = len(layers)
        self.layers = layers

    def stochastic_gradient_descend(self, training_data, epoch, batch_size, learning_rate, test_data):
        for i in range(epoch):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + batch_size]
                for k in range(0, len(training_data), batch_size)]
            for batch in mini_batches:
                self.update_batch(batch, learning_rate)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    i, self.evaluate(test_data), len(test_data)))

    def update_batch(self, batch, learning_rate):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        for x, y in batch:
            delta_nabla_w, delta_nabla_b = self.backprop(x, y)
            nabla_w = [w + nw for w, nw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [b + nb for b, nb in zip(nabla_b, delta_nabla_b)]
        self.weights = [w - (learning_rate / len(batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (learning_rate / len(batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # activation
        # INPUT layer -- shape (784, 1)
        # LAYER 1 -- shape (30,1)
        # LAYER 2 -- shape (10, 1)
        activation = x

        # contains only x which is feature.
        activations = [activation]
        zs = []

        # This is feed forward
        for w, b in zip(self.weights, self.biases):
            # a0 = LAYER 0 --> INPUT
            # a1 = LAYER 1 --> 30 neurons --> sigmoid( np.dot( w a0 ) + b ) --> np.dot ( w(30,784), a(784,1) ) + b(30,1) == a(30, 1)
            # a3 = LAYER 2 OR result --> 10 neurons --> sigmoid( np.dot( w a2 ) + b ) --> np.dot ( w(10,30), a(30,1) ) + b(10,1) == a(10,1)
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # Feed forward is now done.
        # P2 starts here
        # ZS[-1] is Z3 and is last layer
        # Shape of this derivative (10, 1)
        derivative = 2 * (activations[-1] - y) * sigmoid_derivative(zs[-1])

        # Weights and biases for last layer
        # Bias is basically the derivative for last layer
        # weights in last layer (10,30)
        # derivative . a (last second layer)T
        # NOTE: because we transposed activations when calculating z Transposing back when calculating new weight has no negative effect
        nabla_b[-1] = derivative
        nabla_w[-1] = np.dot(derivative, activations[-2].transpose())

        # Biases formula DeltaB(L) = derivative
        # Weights formula DeltaW(L) = np.dpt(W(L+1), Derivative(L+1)) * a[L] (1 - a(L))
        for l in range(2, self.number_of_layers):
            derivative = np.dot(self.weights[-l + 1].transpose(), derivative) * sigmoid_derivative(zs[-l])
            nabla_b[-l] = derivative
            nabla_w[-l] = np.dot(derivative, activations[-l - 1].transpose())
        return nabla_w, nabla_b

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))


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
    features = [feature/255 for feature in features]

    return list(zip(features, labels))


if __name__ == "__main__":
    main()
