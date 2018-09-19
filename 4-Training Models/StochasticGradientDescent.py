'''
Applying Stochastic Gradient Descent for Linear Regression
'''

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

X = 2 * np.random.rand(100,1)
y = 4 + 3 * X + np.random.randn(100,1)

X_b = np.c_[np.ones((100,1)), X]

eta = .1
m = 100

n_epochs = 50
# Learning schedule hyperparameters
t0, t1 = 5, 50

def learning_schedule(t):
    return t0 / (t + t1)

# Random initialization for theta
theta = np.random.randn(2,1)

for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients
print('Theta: ', theta)

# Using Scikit-Learn
from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(max_iter=50, penalty=None, eta0=.1)
sgd_reg.fit(X, y.ravel())

print('Scikit-Learn: ', sgd_reg.intercept_, sgd_reg.coef_)
