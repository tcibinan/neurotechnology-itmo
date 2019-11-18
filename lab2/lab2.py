import matplotlib.pyplot as plt
import numpy as np

m = 128
delta = 0.25
data = np.loadtxt('../lab1/data/data.txt', delimiter='\t', dtype=int)


def auto_correlation(x, length):
    return np.array([1] + [np.corrcoef(x[:-i], x[i:])[0, 1] for i in range(1, length)])


fig, axes = plt.subplots(1, 2)
axes[0].plot(np.arange(0, m*delta, delta), auto_correlation(data, m))
axes[1].scatter(data[:-1], data[1:])
plt.show()
