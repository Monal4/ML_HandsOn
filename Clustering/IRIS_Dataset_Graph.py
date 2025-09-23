import matplotlib.pyplot as plt
from sklearn import datasets

dataset = datasets.load_iris()
data = dataset.data

plt.figure(figsize=(20, 15))

for i in range(len(data)):
    plt.plot(
        data[i][0],
        data[i][1],
        ['ro', 'go', 'yo'][dataset.target[i]]
    )

plt.show()