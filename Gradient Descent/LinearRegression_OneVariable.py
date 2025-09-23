import math
import random

import numpy as np
import pandas as pd


class GradientDescent:

    def __init__(self, slope, intercept):
        self.slope = slope
        self.intercept = intercept

    def gradient_descent(self, training_set, epoch, batch_size, learning_rate, test_set):
        random.shuffle(training_set)
        for i in range(epoch):
            mini_batches = [training_set[k: k + batch_size] for k in range(len(training_set))]
            for batch in mini_batches:
                self.slope, self.intercept = self.update_mini_batch(batch, learning_rate)
            print("Epoch {0}: {1} / {2}".format(i, self.evaluate(test_set), len(test_set)))
        print('slope: ', self.slope)
        print('intercept: ', self.intercept)

    def update_mini_batch(self, batch, learning_rate):
        delta_slope, delta_intercept = 0, 0

        for x, y in batch:
            derivative_slope, derivative_intercept = self.gradient(x, y)
            delta_slope = delta_slope + derivative_slope
            delta_intercept = delta_intercept + derivative_intercept

        self.slope = self.slope - ((delta_slope / len(batch)) * learning_rate)
        self.intercept = self.intercept - ((delta_intercept / len(batch)) * learning_rate)
        return self.slope, self.intercept

    def gradient(self, x, y):
        y_dash = self.slope * x + self.intercept

        cost_function = (y_dash - y) * (y_dash - y)

        delta_intercept = 2 * (y_dash - y)
        delta_slope = 2 * (y_dash - y) * x
        return delta_slope, delta_intercept

    def evaluate(self, test_set):
        results = [(self.get_y(x), y) for x, y in test_set]
        # x = [x for x,y in results]
        # y = [y for x,y in results]
        # plt.scatter(x, y)
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.title('XY Points Plot')
        # plt.grid(True)
        # plt.show()

        return sum(math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-9) for x, y in results)

    def get_y(self, x):
        return self.slope * x + self.intercept


def main():
    gd = GradientDescent(np.random.randn() * 0.01, np.random.randn() * 0.01)
    dataset = pd.read_csv('Linear Regression - Sheet1.csv')
    x_scaled = (dataset.X - dataset.X.min()) / (dataset.X.max() - dataset.X.min())
    y_scaled = (dataset.Y - dataset.Y.min()) / (dataset.Y.max() - dataset.Y.min())
    dataset = list(zip(x_scaled, y_scaled))
    # dataset = list(zip((x/1000 for x in dataset.X), dataset.Y))
    # dataset = list(zip(dataset.X, dataset.Y))
    random.shuffle(dataset)
    random.shuffle(dataset)
    random.shuffle(dataset)
    random.shuffle(dataset)

    training_set = dataset[:280]
    test_set = dataset[280:]

    gd.gradient_descent(training_set, 30, 20, 0.1, test_set)


if __name__ == "__main__":
    main()
