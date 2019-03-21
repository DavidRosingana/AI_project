import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

data = load_iris()

Y = data["target"]
X = data["data"]

X = X-np.mean(X, axis=0)

k = 2
n = float(X.shape [0])

mu = np.mean(X, axis=0)
cov = np.dot((X-mu).T, (X-mu))/(n-1)

evals, evects = np.linalg.eig(cov)

indices = np.argsort(evals)[::-1]

evals = evals[indices]
evects = evects[indices]

W = np.concatenate([evects [i].reshape(-1,1) for i in range(k)], axis=1)
print(W.shape)

Xpca = np.dot(X, W)
print(Xpca.shape)

plt.scatter(Xpca[:,0], Xpca[:,1], c=Y)
plt.show()