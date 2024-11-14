import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
pd.set_option('display.max_columns', None)
data = pd.read_csv('data/final_df.csv')
#data['Annual_Percent_Change'] = data['GDP'].pct_change(4) * 100
data.set_index(keys = 'Date', inplace=True)
print(data)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['Annual_Percent_Change'], marker='o', color='b', label='Annual GDP Percent Change')

# Adding labels and title
plt.ylabel('GDP Growth')
plt.title('GDP Growth Over Time')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.xticks(fontsize = 8,rotation = 45)

# Display the plot
#plt.show()



result = adfuller(data['Annual_Percent_Change'].dropna())
print(f'p-value: {result[1]}')

if result[1] > 0.05:
    print('Non-stationary')
else:
    print('Stationary')



