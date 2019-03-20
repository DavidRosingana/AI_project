import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

n_samples = 100

X, Y = datasets.make_regression(n_samples=n_samples, n_features=1, n_informative=1, noise=10)

X = np.concatenate([X, np.ones((n_samples, 1))], axis=1)

"""
temp = np.linalg.inv(np.dot(X.T, X))
w = np.dot(np.dot(temp,X.T),Y)
"""

w,_,_,_ = np.linalg.lstsq(np.dot(X.T, X),np.dot(X.T, Y))

plt.scatter(X,[:,0],Y) #error de sintaxis en :, no funka

ys = np.array([w[0]*x+w[1] for x in range(-3,4,1)])
plt.plot(range(-3,4,1), ys, c="r")

plt.show()