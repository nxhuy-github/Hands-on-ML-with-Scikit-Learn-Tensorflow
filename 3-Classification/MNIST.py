from sklearn.datasets import fetch_mldata
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

mnist = fetch_mldata('MNIST original')

X, y = mnist['data'], mnist['target']

# splitting the dataset into training set & testing set
X_train, y_train, X_test, y_test = X[:60000], y[:60000], X[60000:], y[60000:]

# shuffling the dataset
suffle_index = np.random.permutation(60000)
X_train, y_train = X_train[suffle_index], y_train[suffle_index]

# Binary Classifier
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)


print(sgd_clf.predict([X[0]]))
print(sgd_clf.predict([X[36000]]))

