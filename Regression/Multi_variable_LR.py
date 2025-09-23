import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import linear_model

# CONSOLE LOGGING RULE
pd.set_option("display.max_rows", 1000, "display.min_rows", 200, "display.max_columns", None, "display.width", None)

# IMPORT DATASET
mobile_data = pd.read_csv('Mobile Price Prediction Datatset.csv')
print(mobile_data.head(10))
print(' \n Total number of Records', len(mobile_data))

# CHECK NAN and FILL
print('\n \n NAN\'s', mobile_data.isna().sum())
mobile_data.fillna({
        'Ratings': mobile_data['Ratings'].mean(),
        'RAM': mobile_data['RAM'].mean(),
        'ROM': mobile_data['ROM'].mean(),
        'Mobile_Size': mobile_data['Mobile_Size'].mean(),
        'Selfi_Cam': mobile_data['Selfi_Cam'].mean()
    }, inplace=True)
print('\n \n NAN\'s after fillna', mobile_data.isna().sum())

feature_data = mobile_data[['Ratings', 'RAM', 'ROM', 'Mobile_Size', 'Primary_Cam', 'Selfi_Cam', 'Battery_Power']]
print('\n', feature_data.head(10))

# feature_data.plot(y='Price')
# plt.show()

# POPULATE LABEL
# feature_data['Price Label'] = feature_data['Price']
# print('\n', feature_data.head(10))

# Create feature and label
label = np.array(mobile_data['Price'])
features = np.array(feature_data)
print('\n', features[1:5])

scaled_features = preprocessing.scale(features)
print('\n', scaled_features[750:753])

(feature_train, feature_test, label_train, label_test) = model_selection.train_test_split(scaled_features, label, test_size=0.1)
model = linear_model.LinearRegression()

model.fit(feature_train, label_train)
# print(feature_test)
label_predicted = model.predict(feature_test)
plt.scatter(label_test, label_predicted)
plt.show()
print(model.score(feature_test, label_test))