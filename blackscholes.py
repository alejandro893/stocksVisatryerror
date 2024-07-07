# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 18:49:58 2024

@author: Alejandro
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D

# Black-Scholes model for call option price
def black_scholes_call(S, X, T, r, sigma):
    d1 = (np.log(S / X) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - X * np.exp(-r * T) * norm.cdf(d2)

# Generate synthetic data
np.random.seed(42)
S0 = 100  # current stock price
r = 0.01  # risk-free rate
vol_surface = []

strike_prices = np.linspace(80, 120, 10)
maturities = np.linspace(0.1, 2, 10)

for X in strike_prices:
    for T in maturities:
        # Generate random volatility
        sigma = 0.2 + 0.1 * np.random.randn()
        # Calculate option price
        price = black_scholes_call(S0, X, T, r, sigma)
        vol_surface.append((X, T, sigma, price))

# Calculate implied volatility
def implied_volatility(price, S, X, T, r):
    # Objective function: minimize the squared difference between market price and BS price
    def objective_function(sigma):
        return (black_scholes_call(S, X, T, r, sigma) - price) ** 2

    # Initial guess for volatility
    result = minimize(objective_function, 0.2, bounds=[(0.001, 2.0)], method='L-BFGS-B')
    return result.x[0]

implied_vols = []
for X, T, _, price in vol_surface:
    imp_vol = implied_volatility(price, S0, X, T, r)
    implied_vols.append((X, T, imp_vol))

# Convert to numpy arrays for plotting
strike_prices = np.array([x for x, t, iv in implied_vols])
maturities = np.array([t for x, t, iv in implied_vols])
implied_vols = np.array([iv for x, t, iv in implied_vols])

# Plot the volatility surface
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(strike_prices, maturities, implied_vols, cmap='viridis')
ax.set_title('Implied Volatility Surface')
ax.set_xlabel('Strike Price')
ax.set_ylabel('Time to Maturity')
ax.set_zlabel('Implied Volatility')
plt.show()
