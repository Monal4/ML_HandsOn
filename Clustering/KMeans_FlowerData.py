from sklearn import datasets, cluster
import matplotlib.pyplot as plt

dataset = datasets.load_iris()
print(dataset.keys())
data = dataset.data
print(data[0:5])

KMeansCluster = cluster.KMeans(n_clusters=3)
KMeansCluster.fit(data)

print(KMeansCluster.cluster_centers_)

print('Actual: ', dataset.target[0])
print('Prediction: ', KMeansCluster.predict([data[0]]))

plt.figure(figsize=(20, 15))

plt.scatter(
    KMeansCluster.cluster_centers_[:, 0],
    KMeansCluster.cluster_centers_[:, 1]
)

for i in range(len(data)):
    plt.plot(
        data[i][0],
        data[i][1],
        ['ro', 'go', 'yo'][KMeansCluster.labels_[i]]
    )


plt.show()
