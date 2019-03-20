import numpy as np
from numpy.linalg import *
import matplotlib.pyplot as plt

X = np.random.normal (size =(2, 100))
plt.scatter (X[0],X[1])
plt.show()


Y = np.array([[3, 3], [2., 5]])
Z = np.asarray([[3, 3], [2, 5]])

print(Y)
print(Z)
X_inv = inv(Y)
print(X_inv)
