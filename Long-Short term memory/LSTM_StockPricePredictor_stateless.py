import pandas as pd
from decimal import Decimal
from sklearn import preprocessing
import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    return 1 - np.tanh(x) ** 2


def softmax(z):
    exp = np.exp(z - np.max(z))
    return exp / np.sum(exp)


class Network:
    def __init__(self):
        # total_cell_states = 32
        #   weight_shapes = [(total_cell_states, 4), (total_cell_states, total_cell_states), (total_cell_states, 4),
        #    (total_cell_states, total_cell_states), (total_cell_states, 4), (total_cell_states, total_cell_states),
        #     (total_cell_states, 4), (total_cell_states, total_cell_states), (2, total_cell_states)]
        weight_shapes = [(32, 4), (32, 32), (32, 4), (32, 32), (32, 4), (32, 32), (32, 4), (32, 32), (2, 32)]
        self.weights = [np.random.randn(shape[0], shape[1]) for shape in weight_shapes]
        bias_shapes = [(32, 1), (32, 1), (32, 1), (32, 1), (2, 1)]
        self.biases = [np.random.randn(shape[0], shape[1]) for shape in bias_shapes]

    def LSTM(self, training_features, learning_rate, batch_size, testing_features):
        batches = []
        # print("weights before : ", self.weights)
        # print("Biases before : ", self.biases)
        for feature in range(0, len(training_features) - batch_size):
            self.create_batch(batch_size, batches, feature, training_features)

        counter = 0
        for cells, result in batches:
            biases_backprop, weights_backprop = self.feed_forward_and_back_prop_batch(
                cells, result)
            counter += 1
            self.weights = [(w - (learning_rate * dw) / 60) for w, dw in zip(self.weights, weights_backprop)]
            self.biases = [(b - (learning_rate * db) / 60) for b, db in zip(self.biases, biases_backprop)]
            if counter == 10979:
                # print("weights after : ", self.weights)
                # print("Biases after : ", self.biases)
                print("Break point")

    def feed_forward_and_back_prop_batch(self, cells, result):
        weights_backprop = [np.zeros(weight.shape) for weight in self.weights]
        biases_backprop = [np.zeros(bias.shape) for bias in self.biases]
        activations, final_activation, z1s, z2s, z3s, z4s, cell_states = self.feed_forward(cells)

        self.backprop(activations, biases_backprop, cell_states, cells,
                                                                   final_activation, result, weights_backprop, z1s, z2s,
                                                                   z3s, z4s)
        return biases_backprop, weights_backprop

    def backprop(self, activations, biases_backprop, cell_states, cells, final_activation, result, weights_backprop,
                 z1s, z2s, z3s, z4s):
        derivative_cross_entropy = final_activation - result  # delta0 = shape (2,1)
        # Update NN weights and biases
        biases_backprop[-1] = derivative_cross_entropy
        weights_backprop[-1] = np.dot(derivative_cross_entropy, activations[-1].transpose())

        self.update_lstm_cell_weights_and_biases(activations, biases_backprop, cell_states, cells,
                                                 derivative_cross_entropy, weights_backprop, z1s, z2s, z3s, z4s)
        # compute backprop hidden and cell states
        # backprop_cell_state, backprop_hidden_state = self.get_backprop_initial_hidden_and_cell_state(cell_states,
        #                                                                                              derivative_cross_entropy,
        #                                                                                              z1s, z2s, z3s,
        #                                                                                              z4s)
        # return backprop_cell_state, backprop_hidden_state

    def get_backprop_initial_hidden_and_cell_state(self, cell_states, derivative_cross_entropy, z1s, z2s, z3s, z4s):
        backprop_hidden_state = np.dot(self.weights[-1].transpose(), derivative_cross_entropy) * (
                sigmoid_prime(z4s[0]) * tanh(cell_states[1])
                +
                (
                        (sigmoid(z4s[0]) * tanh_prime(cell_states[1])) *
                        (
                                cell_states[0] * sigmoid_prime(z1s[0])
                                + (sigmoid_prime((z2s[0])) * tanh(z3s[0]))
                                + (sigmoid(z2s[0]) * tanh_prime(z3s[0]))
                        )
                )
        )
        backprop_cell_state = np.dot(self.weights[-1].transpose(), derivative_cross_entropy) * (
                sigmoid(z4s[0]) * tanh_prime(cell_states[1]) * sigmoid(z1s[0])
        )
        return backprop_cell_state, backprop_hidden_state

    def update_lstm_cell_weights_and_biases(self, activations, biases_backprop, cell_states, cells,
                                            derivative_cross_entropy, weights_backprop, z1s, z2s, z3s, z4s):
        # Never transpose deltas/derivatives
        for index in range(1, 60):
            #  Back prop O1 -- Delta 2
            delta_2 = np.dot(self.weights[-1].transpose(), derivative_cross_entropy) * sigmoid_prime(
                z4s[-index]) * tanh(cell_states[-index])
            weights_backprop[-2] += np.dot(delta_2, activations[-index - 1].transpose())
            weights_backprop[-3] += np.dot(delta_2, cells[-index].transpose())
            biases_backprop[-2] += delta_2

            # Back prop O2 -- Delta 1 -- dl/dci
            delta_1 = np.dot(self.weights[-1].transpose(), derivative_cross_entropy) * sigmoid(
                z4s[-index]) * tanh_prime(cell_states[-index])

            # Back prop 3 - Delta 3 and delta 4 -- derived from dl/dci
            delta_3 = delta_1 * sigmoid(z2s[-index]) * tanh_prime(z3s[-index])
            weights_backprop[-4] += np.dot(delta_3, activations[-index - 1].transpose())
            weights_backprop[-5] += np.dot(delta_3, cells[-index].transpose())
            biases_backprop[-3] += delta_3

            delta_4 = delta_1 * sigmoid_prime(z2s[-index]) * tanh(z3s[-index])
            weights_backprop[-6] += np.dot(delta_4, activations[-index - 1].transpose())
            weights_backprop[-7] += np.dot(delta_4, cells[-index].transpose())
            biases_backprop[-4] += delta_4

            # Back prop 4 - Delta 5 -- dl/dfi depends on cell_state from last layer
            delta_5 = delta_1 * cell_states[-index - 1]

            # Back prop 5 -- Delta 6
            delta_6 = delta_5 * sigmoid_prime(z1s[-index])
            weights_backprop[-8] += np.dot(delta_6, activations[-index - 1].transpose())
            weights_backprop[-9] += np.dot(delta_6, cells[-index].transpose())
            biases_backprop[-5] += delta_6

    def feed_forward(self, cells):
        cell_states = [np.zeros((32, 1))]
        activations = [np.zeros((32, 1))]
        forget_gates = []
        input_gates = []
        z1s = []
        z2s = []
        z3s = []
        z4s = []
        for cell in cells:
            self.compute_gates(activations, cell, cell_states, forget_gates, input_gates, z1s, z2s, z3s, z4s)
        final_activation = self.softmax_neural_network(activations)
        return activations, final_activation, z1s, z2s, z3s, z4s, cell_states

    def create_batch(self, batch_size, batches, i, training_features):
        cells_for_training = training_features[i:i + batch_size - 1]
        testing_cell = training_features[i + batch_size]
        result = np.array([(testing_cell[3] < cells_for_training[-1][3]).__int__(),
                           (testing_cell[3] >= cells_for_training[-1][3]).__int__()]).reshape(2, 1)
        batches.append((cells_for_training, result))

    def softmax_neural_network(self, activations):
        # NN for multiple layers
        # for weight, bias in zip(self.weights[7:], self.biases[3:]):
        #     z5 = np.dot(weight, activation) + bias
        #     activation = sigmoid/relu(z5)
        #     activations.append(activation)
        z5 = np.dot(self.weights[8], activations[-1]) + self.biases[4]
        final_activation = softmax(z5)
        return final_activation

    def compute_gates(self, activations, cell, cell_states, forget_gates, input_gates, z1s, z2s, z3s, z4s):
        forget_gate = self.forget_gate(activations, cell, forget_gates, z1s)
        input_gate = self.input_gate(activations, cell, input_gates, z2s, z3s)
        new_cell_state = self.new_long_term_memory(cell_states, forget_gate, input_gate)
        self.new_hidden_state(activations, cell, new_cell_state, z4s)

    def new_hidden_state(self, activations, cell, new_cell_state, z4s):
        z4 = np.dot(self.weights[6], cell) + np.dot(self.weights[7], activations[-1]) + self.biases[3]
        z4s.append(z4)
        O1 = sigmoid(z4)
        O2 = tanh(new_cell_state)
        activation = O1 * O2
        activations.append(activation)

    def new_long_term_memory(self, cell_states, forget_gate, input_gate):
        new_cell_state = cell_states[-1] * forget_gate + input_gate
        cell_states.append(new_cell_state)
        return new_cell_state

    def input_gate(self, activations, cell, input_gates, z2s, z3s):
        z2 = np.dot(self.weights[2], cell) + np.dot(self.weights[3], activations[-1]) + self.biases[1]
        I1 = sigmoid(z2)
        z3 = np.dot(self.weights[4], cell) + np.dot(self.weights[5], activations[-1]) + self.biases[2]
        I2 = tanh(z3)
        input_gate = I1 * I2
        input_gates.append(input_gate)
        z2s.append(z2)
        z3s.append(z3)
        return input_gate

    def forget_gate(self, activations, cell, forget_gates, z1s):
        z1 = np.dot(self.weights[0], cell) + np.dot(self.weights[1], activations[-1]) + self.biases[0]
        forget_gate = sigmoid(z1)
        z1s.append(z1)
        forget_gates.append(forget_gate)
        return forget_gate


def main():
    df = pd.read_csv('AAPL_historical_data.csv', parse_dates=['Date'], dtype=str)
    df = df.sort_values('Date', ascending=True)
    str_data = df[["Open", "High", "Low", "Close"]].values
    data = [[Decimal(value) for value in row] for row in str_data]
    features = preprocessing.StandardScaler().fit_transform(data)
    features = [feature.reshape(4, 1) for feature in features]

    batch_size = 60
    training_features = features[0: len(features) - (len(features) % batch_size)]
    testing_features = features[len(features) - (len(features) % batch_size): len(features)]
    learning_rate = 0.02

    network = Network()
    network.LSTM(training_features, learning_rate, batch_size, testing_features)


if __name__ == "__main__":
    main()
