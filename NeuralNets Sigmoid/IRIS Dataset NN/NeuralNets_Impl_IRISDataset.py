import random

import numpy as np
from sklearn import datasets


class Network(object):

    def __init__(self, layers):
        # NOTE P0
        self.weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]
        self.biases = [np.random.randn(y, 1) for y in layers[1:]]
        self.number_of_layers = len(layers)
        self.layers = layers
        print("Initialized")

    def stochastic_gradient_descend(self, training_data, batch_size, epoch, learning_rate, test_data):
        # NOTE P0
        weights = self.weights
        biases = self.biases

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
            else:
                print
                "Epoch {0} complete".format(j)

    def update_batch(self, batch, learning_rate):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        for x, y in batch:
            delta_nabla_w, delta_nabla_b = self.backprop(x, y)
            nabla_w = [w + nw for w, nw in zip(self.weights, delta_nabla_w)]
            nabla_b = [b + nb for b, nb in zip(self.biases, delta_nabla_b)]
        self.weights = [w - (learning_rate / len(batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (learning_rate / len(batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # print("P0 finished")
        # NOTE P0 ends here
        # For feed forward we need to have activation for each neuron
        # NOTE P1 starts here

        # activation
        # INPUT layer -- shape (1, 4) -- reshape (4, 1)
        # LAYER 1 -- shape (5,1)
        # LAYER 2 -- shape (2, 1)
        # LAYER 3 -- shape (3, 1)
        activation = x
        activation = activation.reshape(-1, 1)

        # contains only x which is feature
        activations = [activation]
        zs = []

        for w, b in zip(self.weights, self.biases):
            # LAYER 0 -- INPUT
            # LAYER 1 -- 5 neurons -- np.dot( w LAYER 0 ) + b -- np.dot ( w[(5,4)], a[(4,1)] ) + b[(5,1)]
            # LAYER 2 -- 2 neurons -- np.dot( w LAYER 1 ) + b -- np.dot ( w[(2,5)], a[(5,1)] ) + b[(2,1)]
            # LAYER 3 OR result -- 3 neurons -- np.dot( w LAYER 2 ) + b -- np.dot ( w[(3,2)], a[(2,1)] ) + b[(3,1)]
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # print("P1 finished")

        # Feed forward is now done. P1 FINISHES HERE
        # P2 starts here
        # ZS[-1] is Z3 and is last layer
        derivative = (activations[-1] - y) * sigmoid_derivative(zs[-1])

        # Weights and biases for last layer
        # Bias is basically the derivative for last layer
        # weights in last layer (3,2)
        # derivative . a (last second layer)T
        # NOTE: because we transposed activations when calculating z Transposing back when calculating new weight has no negative effect
        nabla_b[-1] = derivative
        nabla_w[-1] = np.dot(derivative, activations[-2].transpose())

        for l in range(2, self.number_of_layers):
            derivative = np.dot(self.weights[-l + 1].transpose(), derivative) * sigmoid_derivative(zs[-l])
            nabla_b[-l] = derivative
            nabla_w[-l] = np.dot(derivative, activations[-l - 1].transpose())
        return nabla_w, nabla_b

    def evaluate(self, test_data):
        y_dash = [np.argmax(y) for x, y in test_data]
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for ((x, z), y) in zip(test_results, y_dash))

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            reshape = a.reshape(-1, 1)
            a = sigmoid(np.dot(w, reshape) + b)
        return a


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))


def main():
    # NOTE P0 Starts
    net = Network([4, 5, 2, 3])
    label_data = datasets.load_iris().target
    flower_dataset = [dataset/10 for dataset in datasets.load_iris().data]

    label_data_vector = [np.zeros((3, 1)) for y in label_data]

    for index in range(len(label_data_vector)):
        label = label_data_vector[index]
        label[label_data[index]] = 1
        label_data_vector[index] = label

    training_data = list(zip(flower_dataset, label_data_vector))
    random.shuffle(training_data)
    random.shuffle(training_data)
    random.shuffle(training_data)
    test_data = training_data[135:]
    net.stochastic_gradient_descend(training_data, 5, 30, 2, test_data)


if __name__ == "__main__":
    main()
