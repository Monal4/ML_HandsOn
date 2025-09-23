import pandas as pd
from sklearn import preprocessing, cluster, decomposition
import matplotlib.pyplot as plt

pd.set_option("display.max_rows", 1000, "display.min_rows", 200, "display.max_columns", None, "display.width", None)

dataset = pd.read_csv('CC GENERAL 2.csv', sep=',')
print(dataset.columns)
print(dataset.head(10))
print(dataset.describe())
dataset.fillna(0, inplace=True)

feature_dataset = dataset.drop('CUST_ID', axis=1)
scaled_features = preprocessing.StandardScaler().fit_transform(feature_dataset)

pca = decomposition.PCA(n_components=2)

pca_features = pca.fit_transform(scaled_features)

model = cluster.KMeans(n_clusters=2)
model.fit(pca_features)
print(model.cluster_centers_)

plt.scatter(
    model.cluster_centers_[:, 0],
    model.cluster_centers_[:, 1]
)

for i in range(len(pca_features)):
    plt.plot(
        pca_features[i][0],
        pca_features[i][1],
        ['r', 'g', 'y', 'o', 'b'][model.labels_[i]]
    )


plt.show()
