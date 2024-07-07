# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 15:21:05 2024

@author: Alejandro
"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Step 1: Generate synthetic data
np.random.seed(42)
frequency_data = np.random.poisson(lam=5, size=100)  # 10 years of data with an average of 5 events per year
severity_data = np.random.lognormal(mean=10, sigma=1, size=500)  # 50 loss events

# Plot the empirical distributions of the frequency and severity
plt.figure()
plt.hist(frequency_data, bins=50, density=True, alpha=0.75)
plt.show()
plt.figure()
plt.hist(severity_data, bins=50, density=True, alpha=0.75)
plt.show()

# Step 2: Fit statistical distributions
frequency_lambda = np.mean(frequency_data)
shape, loc, scale = stats.lognorm.fit(severity_data, floc=0)

# Fit Gamma distribution for severity
# severity_shape, severity_loc, severity_scale = gamma.fit(df['severity'], floc=0)
# Fit Negative Binomial distribution for frequency
#freq_mean, freq_lambda = df['frequency'].mean()
#freq_var = df['frequency'].var()
#freq_p = freq_mean / freq_var
#freq_n = freq_mean * freq_p / (1 - freq_p)
#poisson_pmf = poisson.pmf(x, freq_lambda)
#neg_binom_pmf = nbinom.pmf(x, freq_n, freq_p)

''' prueba de encontrar la frecuencia
# Rename 'types_events' column in df2 to 'event_types'
df2 = df2.rename(columns={'types_events': 'event_types'})

# Concatenate the two dataframes
merged_df = pd.concat([df1, df2], ignore_index=True)

# Extract year from date
merged_df['year'] = merged_df['date'].dt.year

# Calculate frequency of each event type per year
frequency_df = merged_df.groupby(['year', 'event_types']).size().reset_index(name='frequency')

# Pivot the dataframe to have event types as columns
frequency_pivot_df = frequency_df.pivot(index='year', columns='event_types', values='frequency').fillna(0)
'''

print(f"Fitted Poisson lambda (frequency): {frequency_lambda}")
print(f"Fitted Lognormal parameters (severity): shape={shape}, loc={loc}, scale={scale}")

# Step 3: Monte Carlo simulation
num_simulations = 10000
annual_losses = np.zeros(num_simulations)

for i in range(num_simulations):
    num_events = np.random.poisson(lam=frequency_lambda)
    losses = np.random.lognormal(mean=np.log(scale), sigma=shape, size=num_events)
    annual_losses[i] = np.sum(losses)

# Plot the distribution of simulated annual losses
plt.figure()
plt.hist(annual_losses, bins=50, density=True, alpha=0.75)
plt.xlabel('Annual Loss ($)')
plt.ylabel('Probability Density')
plt.title('Distribution of Simulated Annual Losses')
plt.show()

# Step 4: Calculate Value at Risk (VaR)
confidence_level = 0.999
VaR = np.percentile(annual_losses, confidence_level * 100)

print(f"Value at Risk (99.9% confidence level): ${VaR:.2f}")