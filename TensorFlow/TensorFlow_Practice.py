import keras
import pandas as pd
from keras.src.layers import Dense

pd.set_option("display.max_rows", 1000, "display.min_rows", 200, "display.max_columns", None, "display.width", None)

(feature_train, label_train), (feature_test, label_test) = keras.datasets.mnist.load_data()


def flatten(feature):
    uniform_array = [0] * (28 * 28)
    i = 0
    for row in feature:
        for element in row:
            uniform_array[i] = (element / 255)
            i = i + 1
    return uniform_array


transformed_feature_train = [0] * (len(feature_train))
for feature in range(len(feature_train)):
    transformed_feature_train[feature] = flatten(feature_train[feature])

transformed_label_train = [[0] * 10 for _ in range(len(label_train))]

for i in range(len(label_train)):
    transformed_label_train[i][label_train[i]] = 1

model = keras.models.Sequential(
    layers=[
        Dense(30, activation='sigmoid'),
        Dense(10)
    ]
)

model.compile(
    loss=keras.losses.sparse_categorical_crossentropy,
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

transformed_feature_test = [0] * (len(feature_test))
for i in range(len(feature_test)):
    transformed_feature_test[i] = flatten(feature_test[i])

transformed_label_test = [[0] * 10 for _ in range(len(label_test))]

for i in range(len(label_test)):
    transformed_label_test[i][label_test[i]] = 1

model.fit(transformed_feature_train, transformed_label_train, epochs=30, batch_size=10, verbose=2)
model.evaluate(transformed_feature_test, transformed_label_test, batch_size=32, verbose=2)
