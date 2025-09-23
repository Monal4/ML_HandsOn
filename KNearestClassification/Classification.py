import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import neighbors
from matplotlib import pyplot
import joblib

pd.set_option("display.max_rows", 1000, "display.min_rows", 200, "display.max_columns", None, "display.width", None)

data_frame = pd.read_csv('./KNearestClassification/german.data', sep=' ')
print(data_frame.head(15))

# value = np.asarray([data_frame['Age']]).transpose()
# print(value[:10])
# processed = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(value).transpose()
# print(processed)

labels_decoded = {
    'Status': ['A11', 'A12', 'A13', 'A14'],
    'CreditHistory': ['A30', 'A31', 'A32', 'A33', 'A34'],
    'Purpose': ['A40', 'A41', 'A42', 'A43', 'A44', 'A45', 'A46', 'A47', 'A48', 'A49', 'A410'],
    'Savings': ['A61', 'A62', 'A63', 'A64', 'A65'],
    'Employment': ['A71', 'A72', 'A73', 'A74', 'A75'],
    'PersonalStatusAndSex': ['A91', 'A92', 'A93', 'A94', 'A95'],
    'Other': ['A101', 'A102', 'A103'],
    'Property': ['A121', 'A122', 'A123', 'A124'],
    'Installment': ['A141', 'A142', 'A143'],
    'Housing': ['A151', 'A152', 'A153'],
    'Job': ['A171', 'A172', 'A173', 'A174'],
    'ForeignWorker': ['A201', 'A202']
}

labels_encoded = {}

for label in data_frame.columns:
    if labels_decoded.__contains__(label):
        label_encoder = preprocessing.LabelEncoder()
        labels_encoded[label] = label_encoder.fit_transform(labels_decoded.get(label))
        data_frame[label] = label_encoder.fit_transform(data_frame[label])

# data_frame['PersonalStatusAndSex'] = label_encoder.fit_transform(data_frame['PersonalStatusAndSex'])
data_frame.drop(['Telephone'], axis="columns", inplace=True)
print(data_frame.head(10))
print(labels_encoded)

features = np.array(data_frame.drop(['result'], axis=1))
labels = np.array(data_frame['result'])

scaled_features = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(features)
print(scaled_features)

(features_train, features_test, label_train, label_test) = model_selection.train_test_split(scaled_features, labels,
                                                                                            test_size=0.3)

classification_model = neighbors.KNeighborsClassifier(n_neighbors=30)
classification_model.fit(features_train, label_train)

# save model
file_name = 'german_data_score.dat'
joblib.dump(classification_model, file_name)

#  load model
# classification_model = joblib.load(file_name)
# print(classification_model.score(features_test, label_test))

# labels_predicted = classification_model.predict(features_test)

prediction = classification_model.predict(features_test)

print("Score: ", classification_model.score(features_test, label_test))
for i in range(10):
    random_pred = classification_model.predict(features_test)[20+i]
    actual_value = label_test[20+i]
    print("random_pred: ", random_pred, "actual_value: ", actual_value)
