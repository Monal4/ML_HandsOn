import random

import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np


class GradientDescent_MultiVariableLinearRegression:
    def __init__(self, numberOfFeatures):
        self.numberOfFeatures = numberOfFeatures
        self.slopes = [np.random.randn() * 0.01] * numberOfFeatures
        self.intercept = np.random.randn() * 0.01

    def gradient_descent(self, training_set, epoch, mini_batch_size, learning_rate, test_set):
        for i in range(epoch):
            random.shuffle(training_set)
            mini_batches = [training_set[k: k + mini_batch_size] for k in range(0, len(training_set))]
            for batch in mini_batches:
                self.update_mini_batches(batch, learning_rate)
            print("Epoch {0}: {1} / {2}".format(
                i, self.evaluate(test_set), len(test_set)))

    def update_mini_batches(self, batch, learning_rate):
        slopes_sum = [0] * len(self.slopes)
        intercept_sum = 0

        for x, y in batch:
            derived_slope, derived_intercept = self.backprop(x, y)
            slopes_sum = [s1 + s2 for s1, s2 in zip(derived_slope, slopes_sum)]
            intercept_sum = intercept_sum + derived_intercept
        self.slopes = [s1 - ((s2 / len(batch)) * learning_rate) for s1, s2 in zip(self.slopes, slopes_sum)]
        self.intercept = self.intercept - ((intercept_sum/len(batch)) * learning_rate)

    def backprop(self, x, y):
        y_dash = sum(feature * slope for feature, slope in zip(x, self.slopes)) + self.intercept

        derivative_y_dash = 2 * (y_dash - y)

        derived_slopes = [derivative_y_dash * feature_at_i for feature_at_i in x]
        derived_intercept = derivative_y_dash

        return derived_slopes, derived_intercept

    def evaluate(self, test_set):
        y_result = [None] * len(test_set)
        for i in range(len(test_set)):
            features = test_set[i][0]
            actual = test_set[i][1]
            predicted = sum(slope * feature for slope, feature in zip(self.slopes, features)) + self.intercept
            y_result[i] = (predicted, actual)
        difference = [abs(y1 - y2) for y1, y2 in y_result]
        exact_matches = sum(1 for pred, actual in y_result if abs(pred - actual) <= 0.09)

        return exact_matches


def main():
    dataset = pd.read_csv('Student_Performance.csv')
    dataset = pd.get_dummies(dataset, columns=['Extracurricular Activities'], dtype='int')
    scaler = StandardScaler()

    scaled_dataset = scaler.fit_transform(dataset)
    scaled_dataset_df = pd.DataFrame(scaled_dataset, columns=dataset.columns)

    # feature_train = scaled_dataset_df[:9500].drop('Performance Index', axis=1)
    # label_train = scaled_dataset_df['Performance Index'][:9500]
    #
    # feature_test = scaled_dataset_df[9500:].drop('Performance Index', axis=1)
    # label_test = scaled_dataset_df['Performance Index'][9500:]

    feature_train = scaled_dataset_df[['Hours Studied', 'Previous Scores']][:9500]
    label_train = scaled_dataset_df['Performance Index'][:9500]

    feature_test = scaled_dataset_df[['Hours Studied', 'Previous Scores']][9500:]
    label_test = scaled_dataset_df['Performance Index'][9500:]

    feature_set = list(zip(feature_train.values, label_train.values))
    test_set = list(zip(feature_test.values, label_test.values))

    gd = GradientDescent_MultiVariableLinearRegression(len(feature_train.columns))
    gd.gradient_descent(feature_set, 300, 10, 0.01, test_set)


if __name__ == "__main__":
    main()
