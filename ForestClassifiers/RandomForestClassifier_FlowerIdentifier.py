import numpy as np
from IPython.core.display_functions import display
from sklearn import datasets, model_selection, ensemble, metrics
from sklearn.tree import export_graphviz
import graphviz

dataframe = datasets.load_iris()
print(dataframe.keys())
print(dataframe.feature_names)
print(dataframe.data[:5], dataframe.target_names)

features = dataframe.data
labels = dataframe.target

feature_train, feature_test, label_train, label_test = model_selection.train_test_split(features, labels, test_size=0.2)

random_forest_classifier = ensemble.RandomForestClassifier(n_estimators=50)
random_forest_classifier.fit(feature_train, label_train)

print("\n Confusion Matrix:\n",
      metrics.confusion_matrix(label_test, random_forest_classifier.predict(feature_test), labels=[0, 1, 2]))
print("\n Model score:", random_forest_classifier.score(feature_test, label_test))
print("\n Classification report:\n",
      metrics.classification_report(label_test, random_forest_classifier.predict(feature_test), labels=[0, 1, 2]))

# for i in range(10):
#     tree = random_forest_classifier.estimators_[i]
#     dot_data = export_graphviz(tree,
#                                feature_names=dataframe.feature_names,
#                                filled=True,
#                                max_depth=2,
#                                impurity=False,
#                                proportion=True)
#     graph = graphviz.Source(dot_data)
#     display(graph)
