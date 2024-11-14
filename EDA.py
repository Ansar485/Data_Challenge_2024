import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
pd.set_option('display.max_columns', None)

data = pd.read_csv('data/final_df.csv')
data.set_index(keys = 'Date', inplace=True)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['GDP'], marker='o', color='b', label='Nominal GDP')

# Adding labels and title
plt.ylabel('GDP')
plt.title('GDP Over Time')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.xticks(fontsize = 8,rotation = 45)

# Display the plot
plt.show()

# Augmented Dickey-Fuller test for Stationarity
result = adfuller(data['GDP'].dropna())
print(f'p-value: {result[1]}')

if result[1] > 0.05:
    print('Non-stationary')
else:
    print('Stationary')



# Autocorrelation plot
plt.figure(figsize=(10, 6))
plot_acf(data['GDP'], lags = 12, title = "Autocorrelation of Quarterly GDP")
plt.xlabel('Lag (Quarters)')
plt.ylabel('Autocorrelation')
plt.grid(True)
plt.tight_layout()
plt.show()

# We can see that our data is non-stationary and has statistically significant autocorrelation at lag = 4 (indicating yearly correlation)