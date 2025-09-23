import numpy as np
import pandas as pd
from sklearn import cluster, preprocessing, decomposition
import matplotlib.pyplot as plt

pd.set_option("display.max_rows", 1000, "display.min_rows", 200, "display.max_columns", None, "display.width", None)
wine_dataset = pd.read_csv('wine-clustering.csv', sep=',')

print(wine_dataset.columns)
print(wine_dataset.describe())

scaled_features_dataset = wine_dataset
scaled_features_dataset[wine_dataset.columns] = preprocessing.StandardScaler().fit_transform(scaled_features_dataset)

pca = decomposition.PCA(2)

pca_features = pca.fit_transform(scaled_features_dataset)
print(pd.DataFrame(pca_features).describe())


KMeansCluster = cluster.KMeans(n_clusters=3)
KMeansCluster.fit(pca_features)
print(KMeansCluster.cluster_centers_)

plt.scatter(
    KMeansCluster.cluster_centers_[:, 0],
    KMeansCluster.cluster_centers_[:, 1]
)

for i in range(len(pca_features)):
    plt.plot(
        pca_features[i][0],
        pca_features[i][1],
        ['ro', 'go', 'yo'][KMeansCluster.labels_[i]]
    )


plt.show()
