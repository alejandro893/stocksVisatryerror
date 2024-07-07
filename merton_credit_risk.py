# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 17:19:11 2024

@author: Alejandro
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 20:52:53 2024

@author: Alejandro
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, t
import matplotlib.pyplot as plt

# Parameters
assets = {
    'car': 30e6,     # $30 million
    'pharma': 20e6,  # $20 million
    'treasury': 50e6 # $50 million
}
total_assets = sum(assets.values())
debt_value = 80e6             # $80 million
time_to_maturity = 1          # 1 year

# Sector-specific interest rates (annual)
interest_rates = {
    'car': 0.06,     # 6%
    'pharma': 0.05,  # 5%
    'treasury': 0.03 # 3%
}

# Risk-free rate (EURIBOR)
risk_free_rate = 0.01  # 1%

# Step 1: Generate synthetic historical data for each sector
np.random.seed(42)
num_days = 265  # Business days in a year

# Simulate daily returns
def generate_daily_returns(mean, std, size):
    return np.random.normal(loc=mean, scale=std, size=size)

# Car industry
car_returns = generate_daily_returns(0.0002, 0.02, num_days)
car_values = assets['car'] * np.exp(np.cumsum(car_returns))

# Pharma industry
pharma_returns = generate_daily_returns(0.00015, 0.015, num_days)
pharma_values = assets['pharma'] * np.exp(np.cumsum(pharma_returns))

# Treasury bonds
treasury_returns = generate_daily_returns(0.0001, 0.01, num_days)
treasury_values = assets['treasury'] * np.exp(np.cumsum(treasury_returns))

# Combine historical values into a DataFrame
historical_values = pd.DataFrame({
    'car': car_values,
    'pharma': pharma_values,
    'treasury': treasury_values
})

# Plot the historical asset values
historical_values.plot()
plt.xlabel('Days')
plt.ylabel('Asset Value ($)')
plt.title('Historical Asset Values by Sector')
plt.show()

# Step 2: Calculate daily log returns and estimate annualized volatilities
log_returns = np.log(historical_values / historical_values.shift(1)).dropna()

# Annualized volatilities
annualized_volatilities = log_returns.std() * np.sqrt(num_days)
print("Annualized Volatilities:\n", annualized_volatilities)

# Step 3: Calculate correlation matrix
correlation_matrix = log_returns.corr()
print("Correlation Matrix:\n", correlation_matrix)

# Step 4: Simulate multivariate t-distribution
# Degrees of freedom for t-distribution
df = 5

# Generate random samples from multivariate t-distribution
mean_returns = log_returns.mean().values
cov_matrix = log_returns.cov().values
samples = np.random.multivariate_normal(mean_returns, cov_matrix, num_days)

t_samples = t(df).rvs(size=(num_days, 3))
multivariate_t_samples = mean_returns + np.dot(t_samples, np.linalg.cholesky(cov_matrix).T)

# Step 5: Merton model calculations
total_asset_values = historical_values.sum(axis=1).values[-1]

d1 = (np.log(total_asset_values / debt_value) + (risk_free_rate + 0.5 * annualized_volatilities.mean()**2) * time_to_maturity) / (annualized_volatilities.mean() * np.sqrt(time_to_maturity))
d2 = d1 - annualized_volatilities.mean() * np.sqrt(time_to_maturity)

# Equity value (as a call option)
equity_value = total_asset_values * norm.cdf(d1) - debt_value * np.exp(-risk_free_rate * time_to_maturity) * norm.cdf(d2)

# Probability of default (PD)
pd = norm.cdf(-d2)

# Distance to Default (DD)
dd = d1

# Print results
print(f"Current Total Asset Value: ${total_asset_values:.2f}")
print(f"Debt Value: ${debt_value:.2f}")
print(f"Annualized Volatility (Mean): {annualized_volatilities.mean():.4f}")
print(f"Equity Value: ${equity_value:.2f}")
print(f"Probability of Default: {pd:.4f}")
print(f"Distance to Default: {dd:.2f}")