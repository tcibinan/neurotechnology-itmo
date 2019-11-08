import matplotlib.pyplot as plt
import numpy as np


m = 128
delta = 250
data = np.loadtxt('../lab1/data/data.txt', delimiter='\t', encoding="utf-16")

const = (m*sum(data[i]**2 for i in range(1, m+1))) - (sum(data[1:m]))**2
coefs = []

for k in range(m):
    a = m*sum(data[i]*data[i+k] for i in range(1, m+1))
    b = (sum(data[1:m]))*(sum(data[i+k] for i in range(1, m+1)))
    c = m*(sum(data[i]**2 for i in range(k+1, k+m+1))) - (sum(data[k+1:k+m]))**2
    r = (a-b)/np.sqrt(const*c)
    coefs.append(r)

plt.plot(range(m), coefs)
plt.show()
