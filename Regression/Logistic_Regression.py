import numpy as np
from sklearn import neighbors, linear_model, preprocessing, model_selection, metrics
import pandas as pd

pd.set_option("display.max_rows", 1000, "display.min_rows", 200, "display.max_columns", None, "display.width", None)

data_frame = pd.read_csv('diabetes.csv', sep=',')
features = np.array(data_frame.drop('Outcome', axis=1))
label = np.array(data_frame['Outcome'])

scaled_features = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(features)

classification_model = neighbors.KNeighborsClassifier(n_neighbors=15)
logistic_regression_model = linear_model.LogisticRegression()

(feature_train, feature_test, label_train, label_test) = model_selection.train_test_split(scaled_features, label,
                                                                                          test_size=0.15)

classification_model.fit(feature_train, label_train)

# predicted_values = [0] * 100
# actual_values_classification = [0] * 100

# CLASSIFICATION PREDICTION
# for i in range(100):
#     predicted_values[i] = classification_model.predict(feature_test)[i]
#     actual_values_classification[i] = label_test[i]
#
# print(metrics.classification_report(actual_values_classification, predicted_values))

#  LOGISTIC PREDICTION
logistic_regression_model.fit(feature_train, label_train)
#
# for i in range(100):
#     predicted_values[i] = logistic_regression_model.predict(feature_test)[i]
#     actual_values_classification[i] = label_test[i]
#
# print(metrics.classification_report(actual_values_classification, predicted_values))

# Confusion Matrix

Logistic_CF = metrics.confusion_matrix(label_test, logistic_regression_model.predict(feature_test), labels=[0, 1])
Classification_CF = metrics.confusion_matrix(label_test, classification_model.predict(feature_test))

print('\n logistic_regression_model score', logistic_regression_model.score(feature_test, label_test))
print("\n Logistic Confusion Matrix \n", Logistic_CF)

print('\n Classification score', classification_model.score(feature_test, label_test))
print("\n KNearestNeighbor Classification Confusion Matrix \n", Classification_CF)
