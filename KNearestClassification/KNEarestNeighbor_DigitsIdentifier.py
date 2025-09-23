import numpy as np
from sklearn import datasets, model_selection, neighbors, preprocessing, metrics
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.max_rows", 1000, "display.min_rows", 200, "display.max_columns", None, "display.width", None)

digits_dataframe = datasets.load_digits()

print(digits_dataframe.keys())

# plt.gray()
# for i in range(10):
#     plt.matshow(digits_dataframe.images[100+i])
#
# plt.show()

feature_ds = np.array(digits_dataframe.data)
labels = np.array(digits_dataframe.target)

feature_train, feature_test, label_train, label_test = model_selection.train_test_split(feature_ds, labels,
                                                                                        test_size=0.2)

classifier = neighbors.KNeighborsClassifier(n_neighbors=40)

classifier.fit(feature_train, label_train)

print("\n Model Score: ", classifier.score(feature_test, label_test))

print("\n Confusion Matrix \n", metrics.confusion_matrix(label_test, classifier.predict(feature_test)))