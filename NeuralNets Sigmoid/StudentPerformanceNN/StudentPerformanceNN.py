import random
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


def sigmoid(param):
    return 1 / (1 + np.exp(-param))


def sigmoid_prime(param):
    return sigmoid(param) * (1 - sigmoid(param))


class Network:

    def __init__(self, layers):
        self.numOfLayers = len(layers)
        self.weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]
        self.biases = [np.random.randn(y, 1) for y in layers[1:]]

    def gradient_descent(self, training_set, epoch, batch_size, learning_rate, test_set):

        for i in range(epoch):
            random.shuffle(training_set)
            mini_batches = [
                training_set[k: k + batch_size] for k in range(0, len(training_set))
            ]
            for batch in mini_batches:
                self.update_weights_and_biases(batch, learning_rate)
            print("Epoch {0}: {1} / {2}".format(
                i, self.evaluate(test_set), len(test_set)))

    def update_weights_and_biases(self, batch, learning_rate):
        delta_weights = [np.zeros(x.shape) for x in self.weights]
        delta_biases = [np.zeros(x.shape) for x in self.biases]

        for x, y in batch:
            derived_weights, derived_biases = self.backprop(x, y)
            delta_weights = [w + dw for w, dw in zip(delta_weights, derived_weights)]
            delta_biases = [b + db for b, db in zip(delta_biases, derived_biases)]
        self.weights = [w - ((dw / len(batch)) * learning_rate) for w, dw in zip(self.weights, delta_weights)]
        self.biases = [b - ((db / len(batch)) * learning_rate) for b, db in zip(self.biases, delta_biases)]

    def backprop(self, x, y):
        weights_derived = [np.zeros(x.shape) for x in self.weights]
        biases_derived = [np.zeros(x.shape) for x in self.biases]

        activation = x.reshape(-1, 1)
        activations = [activation]
        zs = []

        for weight, bias in zip(self.weights, self.biases):
            z = np.dot(weight, activation) + bias
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        derivative = 2 * (activations[-1] - y) * sigmoid_prime(zs[-1])

        biases_derived[-1] = derivative
        weights_derived[-1] = np.dot(derivative, activations[-2].transpose())

        for index in range(2, self.numOfLayers):
            derivative = np.dot(self.weights[-index + 1].transpose(), derivative) * sigmoid_prime(zs[-index])
            biases_derived[-index] = derivative
            weights_derived[-index] = np.dot(derivative, activations[-index - 1].transpose())

        return weights_derived, biases_derived

    def evaluate(self, test_set):
        results = [(self.feedForward(x.reshape(-1, 1)), y) for x, y in test_set]
        difference = [abs(y1 - y2) for y1, y2 in results]
        exact_matches = sum(1 for pred, actual in results if abs(pred - actual) <= 0.09 )

        return exact_matches

    def feedForward(self, a):
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
        return a


def main():
    dataset = pd.read_csv('Student_Performance.csv')
    dataset = pd.get_dummies(dataset, columns=['Extracurricular Activities'], dtype='int')

    # scaler = StandardScaler()
    scaler = MinMaxScaler()

    scaled_dataset = scaler.fit_transform(dataset)
    scaled_dataset_df = pd.DataFrame(scaled_dataset, columns=dataset.columns)

    feature_train = scaled_dataset_df[['Hours Studied', 'Previous Scores']][:9500]
    label_train = scaled_dataset_df['Performance Index'][:9500]

    feature_test = scaled_dataset_df[['Hours Studied', 'Previous Scores']][9500:]
    label_test = scaled_dataset_df['Performance Index'][9500:]

    feature_set = list(zip(feature_train.values, label_train.values))
    test_set = list(zip(feature_test.values, label_test.values))
    correlation_matrix = dataset.corr(numeric_only=True)

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Feature Correlation Heatmap')
    plt.show()


    network = Network([2, 10, 50, 1])
    network.gradient_descent(feature_set, 30, 10, 0.01, test_set)


if __name__ == '__main__':
    main()
