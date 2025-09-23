import numpy as np
import pandas as pd
from sklearn import preprocessing, linear_model, model_selection, metrics, decomposition

pd.set_option("display.max_rows", 1000, "display.min_rows", 200, "display.max_columns", None, "display.width", None)

dataset = pd.read_csv('heart.csv', sep=',')
print(dataset.columns)
print(dataset.head(10))
print(dataset.shape)

label_encoder = preprocessing.LabelEncoder()

label_encoded_feature_dataset = dataset
labels = [
    'Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'
]

for i in range(len(label_encoded_feature_dataset.columns)):
    if labels.__contains__(label_encoded_feature_dataset.columns[i]):
        label_encoded_feature_dataset[label_encoded_feature_dataset.columns[i]] = label_encoder.fit_transform(label_encoded_feature_dataset[label_encoded_feature_dataset.columns[i]])
    else:
        label_encoded_feature_dataset[label_encoded_feature_dataset.columns[i]] = label_encoded_feature_dataset[label_encoded_feature_dataset.columns[i]]

labels = label_encoded_feature_dataset['HeartDisease']
label_encoded_feature_dataset = label_encoded_feature_dataset.drop('HeartDisease', axis=1)

print(label_encoded_feature_dataset.describe())
scaled_features = preprocessing.StandardScaler().fit_transform(label_encoded_feature_dataset)
print(pd.DataFrame(scaled_features).describe())

feature_train, feature_test, label_train, label_test = model_selection.train_test_split(scaled_features, labels, test_size=0.2)

logistic_regression = linear_model.LogisticRegression()
logistic_regression.fit(feature_train, label_train)
print('Score ', logistic_regression.score(feature_test, label_test))

print(
    '\n Confusion Matrix \n', metrics.confusion_matrix(label_test, logistic_regression.predict(feature_test), labels=[0, 1])
)

pca = decomposition.PCA(8)
features_pca = pca.fit_transform(scaled_features)
print(features_pca.shape)
print(pca.explained_variance_ratio_)

feature_train_1, feature_test_1, label_train_1, label_test_1 = model_selection.train_test_split(features_pca, labels, test_size=0.2)

logistic_regression_1 = linear_model.LogisticRegression()
logistic_regression_1.fit(feature_train_1, label_train_1)
print('Score ', logistic_regression_1.score(feature_test_1, label_test_1))
print(
    '\n Confusion Matrix \n', metrics.confusion_matrix(label_test_1, logistic_regression_1.predict(feature_test_1), labels=[0, 1])
)