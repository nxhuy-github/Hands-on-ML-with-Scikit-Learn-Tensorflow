'''
Polynomial Regression
'''

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

m = 100
X = 6 * np.random.rand(m, 1) - 3
y = .5 * X ** 2 + X + 2 + np.random.randn(m, 1)

# Using Scikit-Learn
from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2, include_bias=False)

# First, adding the square of each feature
# X_poly now contains the original feature of x + the square of this feature
X_poly = poly_features.fit_transform(X)
print('X[0]: ', X[0])
print('X_poly[0]: ', X_poly[0])

# Then, applying Linear Regression on this new feature
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
print(lin_reg.intercept_, lin_reg.coef_)

# Visualizing
X_grid = np.arange(min(X), max(X), .1)
X_grid = X_grid.reshape((len(X_grid), 1))

ax = plt.axes()
ax.plot(X, y, 'b.')
ax.plot(X_grid, lin_reg.predict(poly_features.fit_transform(X_grid)), 'r-')
ax.set_xlim([-3, 3])
ax.set_ylim([-3, 10])

plt.show()

