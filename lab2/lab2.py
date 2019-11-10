import matplotlib.pyplot as plt
import numpy as np

m = 128
delta = 250
data = np.loadtxt('../lab1/data/data.txt', delimiter='\t', dtype=int)

rs = []

for k in range(m - 1):
    r = (m * np.sum(data[0:m - 1] * data[k: m - 1 + k]) - np.sum(data[0:m - 1]) * np.sum(data[k: m - 1 + k])) \
        / np.sqrt((m * np.sum(data[0:m - 1] ** 2) - np.sum(data[0:m - 1]) ** 2)
                  * (m * np.sum(data[k:m - 1 + k] ** 2) - np.sum(data[k:m - 1 + k]) ** 2))
    rs.append(r)

fig, axes = plt.subplots(1, 2)
axes[0].plot(range(m - 1), rs)
axes[1].scatter(data[:-1], data[1:])
plt.show()
