from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC

data = load_digits()
Y = data["target"]
X = data["data"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

svmPoly = SVC(kernel="poly", degree=2, coef0=0)
svmPoly.fit(X_train, Y_train)

lsvm = LinearSVC()
lsvm.fit(X_train, Y_train)

print(svmPoly.predict([X_test[0]]))
print(lsvm.predict([X_test[0]]))

print(svmPoly.score(X_test, Y_test))
print(lsvm.score(X_test, Y_test))
