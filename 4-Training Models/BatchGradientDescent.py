'''
Applying Batch Gradient Descent to Linear Regression
'''

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

X = 2 * np.random.rand(100,1)
y = 4 + 3 * X + np.random.randn(100,1)

X_b = np.c_[np.ones((100,1)), X]

# Learning rate
eta = .02
n_iterations = 1000
m = 100

# Data for Predicting
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2,1)), X_new]

# Creating axes to plot
ax = plt.axes()
ax.plot(X, y, 'b.')
ax.set_xlim([0,2])
ax.set_ylim([0,15])

# Random initialization for theta
theta = np.random.randn(2, 1)
for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients
    # Visualizing
    y_predict = X_new_b.dot(theta)
    ax.plot(X_new, y_predict, 'r-')

print('Theta(final): ', theta)
plt.show()
