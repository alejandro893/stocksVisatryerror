# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 19:51:17 2024

@author: Alejandro
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D

class BlackScholesModel:
    def __init__(self, S, X, T, r, sigma):
        self.S = S  # Stock price
        self.X = X  # Strike price
        self.T = T  # Time to maturity
        self.r = r  # Risk-free rate
        self.sigma = sigma  # Volatility
    
    def d1(self):
        return (np.log(self.S / self.X) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
    
    def d2(self):
        return self.d1() - self.sigma * np.sqrt(self.T)
    
    def call_option_price(self):
        return self.S * norm.cdf(self.d1()) - self.X * np.exp(-self.r * self.T) * norm.cdf(self.d2())
    
    @staticmethod
    def implied_volatility(price, S, X, T, r):
        def objective_function(sigma):
            model = BlackScholesModel(S, X, T, r, sigma)
            return (model.call_option_price() - price) ** 2
        
        result = minimize(objective_function, 0.2, bounds=[(0.001, 2.0)], method='L-BFGS-B')
        return result.x[0]

# Generate synthetic data
np.random.seed(42)
S0 = 100  # Current stock price
r = 0.01  # Risk-free rate
vol_surface = []

strike_prices = np.linspace(80, 120, 10)
maturities = np.linspace(0.1, 2, 10)

for X in strike_prices:
    for T in maturities:
        # Generate random volatility
        sigma = 0.2 + 0.1 * np.random.randn()
        # Create Black-Scholes model instance
        bs_model = BlackScholesModel(S0, X, T, r, sigma)
        # Calculate option price
        price = bs_model.call_option_price()
        vol_surface.append((X, T, sigma, price))

# Calculate implied volatility
implied_vols = []
for X, T, _, price in vol_surface:
    imp_vol = BlackScholesModel.implied_volatility(price, S0, X, T, r)
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
