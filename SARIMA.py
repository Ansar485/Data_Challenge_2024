import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
#from pmdarima.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
pd.set_option('display.max_columns', None)

data = pd.read_csv('data/final_df.csv', parse_dates = ['Date'])
data.set_index(keys = 'Date', inplace=True)
data['GDP'] = data['GDP']/1000000 #converting GDP into trillions
data['Yearly_GDP'] = data.groupby('Year')['GDP'].transform('sum') # calculating Yearly GDP


train_data = data.iloc[:31] # 70% of the data was used to train
test_data = data.iloc[31:] # 30% of the date was used to test

gdp_2023 = data.loc['2023-10-01', 'Yearly_GDP']
print(gdp_2023)
doubled_gdp_2023 = 2 * gdp_2023
print(doubled_gdp_2023)

# Use auto_arima to find the best SARIMA model for the training data
stepwise_fit = auto_arima(train_data['GDP'],
                          seasonal=True,
                          m=4,
                          trace=True,
                          error_action='ignore',
                          suppress_warnings=True,
                          stepwise=True)

best_order = stepwise_fit.order
print(f'Best Order: {best_order}')
best_seasonal_order = stepwise_fit.seasonal_order
print(f'Best Seasonal Order: {best_seasonal_order}')
model = SARIMAX(train_data['GDP'],
                order=best_order,
                seasonal_order=best_seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
                trend = 'ct')
results = model.fit()

# Forecast the next observations (matching the length of the test data)
forecast_steps = len(test_data) + 48
forecast = results.get_forecast(steps=forecast_steps)
forecast_values = forecast.predicted_mean
forecast_ci = forecast.conf_int()

yearly_forecast = forecast_values.resample('YE').sum()

# Find the year when the forecasted yearly GDP exceeds the doubled GDP of 2023 Q4
year_reached = yearly_forecast[yearly_forecast >= doubled_gdp_2023].index[0].year

# Display the year when the doubled GDP of 2023 is expected to be reached
print(year_reached)

# Calculate evaluation metrics
mae = mean_absolute_error(test_data['GDP'], forecast_values[:len(test_data)])
mape = mean_absolute_percentage_error(test_data['GDP'], forecast_values[:len(test_data)])
print(f'MAE: {mae}')
print(f'MAPE: {mape}')

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['GDP'], label='Actual GDP', color='black')
plt.plot(train_data.index, train_data['GDP'], label='Training Data', color='blue')
plt.plot(test_data.index, test_data['GDP'], label='Test Data', color='green')
plt.plot(forecast_values.index, forecast_values, label='Forecasted GDP', color='orange')
plt.fill_between(forecast_values.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='orange', alpha=0.2)



# Plot the vertical lines
plt.axvline(x = pd.to_datetime(year_reached, format='%Y'), color='red', linestyle='--', label=f'Year reached: {year_reached}')


# Customize the plot
plt.xlabel('Date')
plt.ylabel('Quarterly GDP (in trillions ₸)')
plt.title('SARIMA Quarterly GDP (in trillions ₸) Forecast with Extended Prediction')
plt.legend()
plt.grid(True)
plt.xticks(fontsize=8, rotation=45)

if not os.path.exists("graphs"):
    os.makedirs("graphs")

plt.savefig('graphs/SARIMA_GDP_Prediction.png')

plt.show()

