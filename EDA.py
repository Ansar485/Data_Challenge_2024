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
plt.plot(data.index, data['GDP'], marker='o', color='b', label='Quarterly GDP')

# Adding labels and title
plt.ylabel('Quarterly GDP')
plt.title('Quarterly GDP Over Time')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.xticks(ticks=range(0, len(data.index), 4), labels=data.index[::4], fontsize=8)

# Display the plot
plt.savefig('graphs/GDP_Current_Trend.png')
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
plt.savefig('graphs/Autocorrelation.png')
plt.show()

# We can see that our GDP data is non-stationary and has statistically significant autocorrelation at lag = 4 (indicating yearly correlation)


# EDA for the GDP divided by economic sectors!


# Load the data from the CSV file
file_path = 'data/ВВП методом производства.csv'  # Replace with your actual file path
data = pd.read_csv(file_path)

# Rename columns for clarity
data.columns = ['Sector'] + [f'Period_{i}' for i in range(1, data.shape[1] - 1)] + ['Unit']
data = data.drop(0)  # Drop the first row if it's a header row

# Melt data into long format
data_long = pd.melt(data, id_vars=['Sector'], var_name='Period', value_name='Value')
data_long['Value'] = pd.to_numeric(data_long['Value'].str.replace(',', ''), errors='coerce')

# Map each Period to a corresponding (Year, Quarter)
start_year = 2010
quarters_per_year = 4
period_mapping = {}
for i in range(1, 56):
    year = start_year + (i - 1) // quarters_per_year
    quarter = ((i - 1) % quarters_per_year) + 1
    period_mapping[f'Period_{i}'] = (year, quarter)

# Apply the mapping and clean up the data
data_long['Year'] = data_long['Period'].map(lambda x: period_mapping.get(x, (None, None))[0])
data_long['Quarter'] = data_long['Period'].map(lambda x: period_mapping.get(x, (None, None))[1])
data_cleaned = data_long.dropna(subset=['Year', 'Quarter', 'Value'])

# Convert Year and Quarter to integers
data_cleaned['Year'] = data_cleaned['Year'].astype(int)
data_cleaned['Quarter'] = data_cleaned['Quarter'].astype(int)

# Reorganize columns
data_cleaned = data_cleaned[['Sector', 'Year', 'Quarter', 'Value']].sort_values(by=['Year', 'Quarter', 'Sector'])

# Save to CSV
output_path = 'Long_Format_GDP_Data.csv'
data_cleaned.to_csv(output_path, index=False)


# Plot 1 for GDP by sectors
data = data_cleaned.copy()

# Filter out the 'GDP' sector since it's the total
data = data[data['Sector'] != 'GDP']

# Calculate total GDP for each Year-Quarter combination
total_gdp = data.groupby(['Year', 'Quarter'])['Value'].sum().reset_index(name='Total_GDP')

# Merge total GDP back with the data to calculate percentage contribution
data = data.merge(total_gdp, on=['Year', 'Quarter'])
data['Contribution_Percentage'] = (data['Value'] / data['Total_GDP']) * 100

# Create a new column for Year-Quarter label for x-axis
data['Year_Quarter'] = data['Year'].astype(str) + ' Q' + data['Quarter'].astype(str)

# Plot quarterly trends for all sectors with distinct colors
plt.figure(figsize=(14, 8))
unique_sectors = data['Sector'].unique()
colors = plt.cm.tab20.colors  # Use a color map with distinct colors

for idx, sector in enumerate(unique_sectors):
    sector_data = data[data['Sector'] == sector]
    plt.plot(sector_data['Year_Quarter'], sector_data['Value'], label=sector, color=colors[idx % len(colors)], marker='o')

plt.title("Quarterly Trends in GDP Contribution for All Sectors with Distinct Colors")
plt.xlabel("Year (Quarter)")
plt.ylabel("GDP Contribution (in monetary units)")
plt.xticks(rotation=90)
plt.legend(title="Sector", bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
plt.tight_layout()
plt.savefig('graphs/GDP_by_sectors_historical_abs.png')
plt.show()


# Plot 2 for GDP by sectors
data = data_cleaned.copy()

# Filter out the 'GDP' sector
data = data[data['Sector'] != 'GDP']

# Calculate total GDP for each Year-Quarter combination
total_gdp = data.groupby(['Year', 'Quarter'])['Value'].sum().reset_index(name='Total_GDP')

# Merge total GDP back with the data to calculate percentage contribution
data = data.merge(total_gdp, on=['Year', 'Quarter'])
data['Contribution_Percentage'] = (data['Value'] / data['Total_GDP']) * 100

# Create a new column for Year-Quarter label for x-axis
data['Year_Quarter'] = data['Year'].astype(str) + ' Q' + data['Quarter'].astype(str)

# Pivot data to have Year_Quarter as rows and Sector as columns for easy stacking
stacked_data = data.pivot_table(index='Year_Quarter', columns='Sector', values='Contribution_Percentage', fill_value=0)

# Plot a wider stacked bar chart without percentage labels on each segment
plt.figure(figsize=(30, 12))
stacked_data.plot(kind='bar', stacked=True, figsize=(30, 12), colormap='tab20')

plt.title("Quarterly Sector Contributions to GDP by Percentage")
plt.xlabel("Year (Quarter)")
plt.ylabel("Contribution to GDP (%)")
plt.xticks(rotation=90)

# Set up legend with smaller font size
plt.legend(title="Sector", bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, fontsize='small', title_fontsize='small')
plt.tight_layout()
plt.savefig('graphs/GDP_by_sectors_historical_share.png')
plt.show()

