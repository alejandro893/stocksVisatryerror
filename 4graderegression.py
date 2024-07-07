# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 17:41:25 2024

@author: Alejandro
"""

import numpy as np
import matplotlib.pyplot as plt

# Generate random data
np.random.seed(42)
x = np.linspace(0, 10, 100)
#y = 3 * np.log(x + 1) + np.random.normal(scale=0.5, size=x.shape) + 0.05 * x**4
y = 3 * np.log(x+0.5) + np.random.normal(scale=0.5, size=x.shape) + 0.05

# Plot the data
plt.figure()
plt.scatter(x, y, label='Data points')
plt.title("Yield Curve Data")
plt.xlabel("Time")
plt.ylabel("Yield")
plt.legend()
plt.show()

# Fit a 4th order polynomial regression model
coefficients = np.polyfit(x, y, 4)
polynomial = np.poly1d(coefficients)

# Generate values for the fitted curve
x_fit = np.linspace(0, 10, 100)
y_fit = polynomial(x_fit)

# Plot the original data and the fitted curve
plt.figure()
plt.scatter(x, y, label='Data points')
plt.plot(x_fit, y_fit, color='red', label='4th Order Regression Fit')
plt.title("4th Order Polynomial Regression Fit")
plt.xlabel("Time")
plt.ylabel("Yield")
plt.legend()
plt.show()
