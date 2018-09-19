'''
Linear Regression using Normal Equation 
'''

import numpy as np

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100,1)

# Adding 1 to each instance of X
X_b = np.c_[np.ones((100,1)), X]
# Calculating the value of Theta that minimizes the cost function
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print('Theta: ', theta_best)
# Making predictions using theta
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2,1)), X_new]
y_predict = X_new_b.dot(theta_best)
print('y predicted: ', y_predict)

import matplotlib
import matplotlib.pyplot as plt

ax = plt.axes()
ax.plot(X, y, 'b.')
ax.plot(X_new, y_predict, 'r')
plt.show()

# Using Scikit-Learn
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
print('Scikit-Learn theta: ', lin_reg.intercept_, lin_reg.coef_)
print('Scikit-Learn y predicted: ', lin_reg.predict(X_new))
