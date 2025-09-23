import random
import os
import numpy as np
from PIL import Image


class Network(object):

    def __init__(self, layers):
        self.weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]
        self.biases = [np.random.randn(y, 1) for y in layers[1:]]
        self.number_of_layers = len(layers)
        self.layers = layers
        print("Initialized")

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
    net = Network([784, 70, 2])

    raw_dataset = os.listdir('simpsons/')
    datasets_features = []
    datasets_labels = []
    for image in raw_dataset:
        image_path = os.path.join('simpsons/', image)
        label = 0
        if(image_path.__contains__('homer')):
            label = 1
        img = Image.open(image_path).convert('L')
        img = img.resize((28, 28))  # example size

        img_array = np.array(img, dtype=np.float32) / 255.0

        img_array = img_array.reshape(28 * 28, 1)
        datasets_features.append(img_array)
        datasets_labels.append(label)

    datasets = list(zip(datasets_features, datasets_labels))

    random.shuffle(datasets)
    random.shuffle(datasets)
    random.shuffle(datasets)
    random.shuffle(datasets)
    random.shuffle(datasets)

    for i in range(0, 254):
        (x,y) = datasets[i]
        label = np.zeros((2, 1))
        label[y] = 1
        datasets[i] = (x, label)

    test_data = datasets[255:]
    ys = []
    for x, y in test_data:
        ys.append(y)
    print(ys)
    net.stochastic_gradient_descend(datasets,30, 5, 0.75, test_data)

if __name__ == "__main__":
    main()
