import keras
import pandas as pd
from keras.src.layers import Dense
from sklearn import datasets
import numpy as np

pd.set_option("display.max_rows", 1000, "display.min_rows", 200, "display.max_columns", None, "display.width", None)

# (feature_train, label_train), (feature_test, label_test) = keras.datasets.mnist.load_data()

feature_train = datasets.load_iris().data
label_train = datasets.load_iris().target

transformed_label_train = np.zeros((150, 3))

for i in range(len(label_train)):
    transformed_label_train[i][label_train[i]] = 1

feature_test = feature_train[135:].astype(np.float32)
label_test = transformed_label_train[135:].astype(np.float32)

feature_train = feature_train[:135].astype(np.float32)
label_train = transformed_label_train[:135].astype(np.float32)

model = keras.models.Sequential(
    layers=[
        Dense(5, activation='sigmoid'),
        Dense(2, activation='sigmoid'),
        Dense(3)
    ]
)

model.compile(
    loss=keras.losses.sparse_categorical_crossentropy,
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)


model.fit(feature_train, datasets.load_iris().target[:135], batch_size=5, epochs=100, verbose=2)
model.evaluate(feature_test, datasets.load_iris().target[135:], batch_size=5, verbose=2)
