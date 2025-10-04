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
        self.weights = [np.random.randn(y, x) * np.sqrt(2.0 / x) for x, y in zip(layers[:-1], layers[1:])]
        self.biases = [np.random.randn(y, 1) for y in layers[1:]]

        self.filter = np.random.randn(3, 3) * np.sqrt(2.0 / 9)
        self.filter_bias = 0

    def CNN(self, training_set, epoch, batch_size, learning_rate, testing_set):
        for i in range(epoch):
            random.shuffle(training_set)
            mini_batches = [
                training_set[k: k + batch_size]
                for k in range(0, len(training_set), batch_size)
            ]
            for batch in mini_batches:
                self.optimize_parameters(batch, learning_rate)
            print("Epoch {0}: {1} / {2}".format(
                i, self.evaluate(testing_set), len(testing_set)))

    def optimize_parameters(self, batch, learning_rate):
        d_filter = np.zeros((3, 3))
        dw_weights = [np.zeros(weight.shape) for weight in self.weights]
        dw_biases = [np.zeros(bias.shape) for bias in self.biases]
        d_filter_bias = 0

        for x, y in batch:
            filter_sample, filter_bias_sample, weights_sample, biases_sample = self.backprop(x, y)
            d_filter += filter_sample
            d_filter_bias += filter_bias_sample
            dw_weights = [(w + dw) for (w, dw) in zip(dw_weights, weights_sample)]
            dw_biases = [(b + db) for (b, db) in zip(dw_biases, biases_sample)]
        self.filter = self.filter - ((learning_rate / len(batch)) * d_filter)
        self.weights = [w - ((learning_rate / len(batch)) * dw) for w, dw in zip(self.weights, dw_weights)]
        self.biases = [b - ((learning_rate / len(batch)) * db) for b, db in zip(self.biases, dw_biases)]
        self.filter_bias = self.filter_bias - (learning_rate / len(batch)) * d_filter_bias
        # print('------')

    def backprop(self, x, y):
        # Feed forward
        activations, d_biases_sample, d_weights_sample, feature_map, pooled_indexes, pooled_map, zs = self.feed_forward(
            x)

        # Back prop and populate d_weights and d_biases for sample
        derivative_cross_entropy = self.back_propagate_NN(activations, d_biases_sample, d_weights_sample, zs, y)

        # Reverse pooled 1D array of (169, 1)
        back_propagated_input = np.dot(self.weights[0].transpose(), derivative_cross_entropy)

        # Back-fill new matrix - reverse feature map
        reverse_feature_map = self.generate_feature_map_from_backprop_derivative(back_propagated_input, feature_map,
                                                                                 pooled_indexes, pooled_map)

        # ReLU prime on back-filled map
        reverse_feature_map *= ReLU_Prime(feature_map)

        # Generate filter from propagated derivative
        d_filter_for_sample = np.zeros((3, 3))

        for u in range(0, 3):
            for v in range(0, 3):
                for i in range(0, 26):
                    for j in range(0, 26):
                        d_filter_for_sample[u, v] += reverse_feature_map[i, j] * x[i + u, j + v]

        filter_bias_sample = np.sum(reverse_feature_map)
        # print('-------')
        return d_filter_for_sample, filter_bias_sample, d_weights_sample, d_biases_sample

    def feed_forward(self, x):
        # feature_map = filter*i + bias == 26*26
        feature_map, feature_map_shape = self.getFeatureMapForSample(x)
        # Relu on feature_map
        feature_map_relu = ReLU(feature_map)
        # pooling through 2*2 matrix == 13*13
        pooled_map, pooled_indexes = self.pool_features(feature_map_relu, feature_map_shape)
        # Neural network
        d_weights_sample = [np.zeros(weight.shape) for weight in self.weights]
        d_biases_sample = [np.zeros(bias.shape) for bias in self.biases]
        # Feed forward sample
        activations, zs = self.feed_forward_pooled_input(pooled_map)
        return activations, d_biases_sample, d_weights_sample, feature_map, pooled_indexes, pooled_map, zs

    def generate_feature_map_from_backprop_derivative(self, back_propagated_input, feature_map, pooled_indexes,
                                                      pooled_map):
        # 26*26 matrix
        reverse_feature_map = np.zeros(feature_map.shape)
        # 169 rows
        pooled_indexes = pooled_indexes.reshape(pooled_map.shape[0] * pooled_map.shape[1], 1)
        for i in range(0, back_propagated_input.shape[0]):
            reverse_feature_map[pooled_indexes[i, 0]] = back_propagated_input[i, 0]
        return reverse_feature_map

    def back_propagate_NN(self, activations, biases_sample, weights_sample, zs, y):
        derivative_cross_entropy = activations[-1] - y
        biases_sample[-1] = derivative_cross_entropy
        weights_sample[-1] = np.dot(derivative_cross_entropy, activations[-2].transpose())
        for i in range(2, self.numberOfLayers):
            derivative_cross_entropy = np.dot(self.weights[-i + 1].transpose(), derivative_cross_entropy) * ReLU_Prime(
                zs[-i])
            biases_sample[-i] = derivative_cross_entropy
            weights_sample[-i] = np.dot(derivative_cross_entropy, activations[-i - 1].transpose())
        return derivative_cross_entropy

    def feed_forward_pooled_input(self, pooled_map):
        features_for_network = pooled_map.reshape(pooled_map.shape[0] * pooled_map.shape[1], 1)
        activation = features_for_network
        activations = [activation]
        zs = []
        for weight, bias in zip(self.weights[0:self.numberOfLayers - 2], self.biases[0:self.numberOfLayers - 2]):
            z = np.dot(weight, activation) + bias
            zs.append(z)
            activation = ReLU(z)
            activations.append(activation)
        z = np.dot(self.weights[-1], activation) + self.biases[-1]
        zs.append(z)
        activation = softmax(z)
        activations.append(activation)
        return activations, zs

    def pool_features(self, feature_map_relu, feature_map_shape):
        stride = 2
        pooled_index = int((feature_map_shape - stride) / stride + 1)
        pooled_map = np.zeros((pooled_index, pooled_index))
        indexes = np.zeros((pooled_index, pooled_index), dtype=tuple)
        for i in range(0, pooled_index):
            for j in range(0, pooled_index):
                i_start = i * 2
                j_start = j * 2

                i_start_stride = i_start + stride
                j_start_stride = j_start + stride
                window = feature_map_relu[i_start:i_start_stride, j_start:j_start_stride]
                value = np.max(window)
                index = np.argmax(window)
                i_index = index // 2
                j_index = index % 2

                pooled_map[i, j] = value

                indexes[i, j] = (i_index + i_start, j_index + j_start)
        return pooled_map, indexes

    def getFeatureMapForSample(self, x):
        feature_shape = x.shape[0]
        filter_shape = self.filter.shape[0]
        feature_map_shape = int((feature_shape - filter_shape) / 1 + 1)
        feature_map = np.zeros((feature_map_shape, feature_map_shape))
        for i in range(0, feature_map_shape):
            for j in range(0, feature_map_shape):
                for u in range(0, filter_shape):
                    for v in range(0, filter_shape):
                        feature_map[i, j] += self.filter[u, v] * x[u + i, v + j]

        feature_map += self.filter_bias
        return feature_map, feature_map_shape

    def evaluate(self, testing_set):
        results = []
        for x, y in testing_set:
            activations, d_biases_sample, d_weights_sample, feature_map, pooled_indexes, pooled_map, zs = self.feed_forward(
                x)
            predicted = np.argmax(activations[-1])
            actual = np.argmax(y)
            results.append((predicted, actual))
        return sum(int(x == y) for (x, y) in results)




def main():
    network = Network([169, 70, 10])
    network = Network([169, 70, 10])

    training_dataset = keras.datasets.mnist.load_data()[0]
    testing_dataset = keras.datasets.mnist.load_data()[1]

    training_dataset_zip = convertLabelsAndZipData(training_dataset, True)
    testing_dataset_zipped = convertLabelsAndZipData(testing_dataset, False)

    network.CNN(training_dataset_zip, 100, 10, 0.001, testing_dataset_zipped)


def convertLabelsAndZipData(dataset, training):
    features = dataset[0]
    labels = dataset[1]
    if training:
        converted_labels = np.zeros((len(labels), 10))

        for i in range(len(converted_labels)):
            converted_labels[i][labels[i]] = 1

        labels = [label.reshape(10, 1) for label in converted_labels]

    # features = [feature.reshape((28 * 28, 1)) for feature in features]
    features = [feature / 255 for feature in features]

    return list(zip(features, labels))


if __name__ == "__main__":
    main()
