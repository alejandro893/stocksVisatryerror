# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 18:03:30 2024

@author: Alejandro
"""

import numpy as np
import matplotlib.pyplot as plt

# Generate random data
np.random.seed(42)
x = np.linspace(0, 10, 100)
y = 2.5 * x + np.random.normal(scale=2, size=x.shape)

# Plot the data
plt.figure()
plt.scatter(x, y, label='Data points')
plt.title("Linear Data")
plt.xlabel("Time")
plt.ylabel("Yield")
plt.legend()
plt.show()


from sklearn.linear_model import LinearRegression

# Reshape x for sklearn
x_reshaped = x.reshape(-1, 1)

# Fit the linear regression model
model = LinearRegression()
model.fit(x_reshaped, y)

# Get the fitted values
y_fit = model.predict(x_reshaped)

# Plot the original data and the fitted line
plt.figure()
plt.scatter(x, y, label='Data points')
plt.plot(x, y_fit, color='red', label='Linear Regression Fit')
plt.title("Linear Regression Fit")
plt.xlabel("Time")
plt.ylabel("Yield")
plt.legend()
plt.show()


import scipy.stats as stats
import statsmodels.api as sm

# Calculate residuals
residuals = y - y_fit

# Plot histogram of residuals
plt.figure()
plt.hist(residuals, bins=30, edgecolor='black')
plt.title("Histogram of Residuals")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()

# Normality tests
print("Anderson-Darling Test:")
ad_test = stats.anderson(residuals)
print(ad_test)

print("\nKolmogorov-Smirnov Test:")
ks_test = stats.kstest(residuals, 'norm', args=(residuals.mean(), residuals.std()))
print(ks_test)

# Q-Q plot
sm.qqplot(residuals, line ='45')
plt.title("Q-Q Plot of Residuals")
plt.show()
