import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define parameters for GDP growth
years = np.arange(2023, 2041)
natural_growth_rate = 0.05  # 5% natural growth
gdp_start = 118.808  # Starting GDP in trillion Tenge
gdp_start_doubled = gdp_start * 2
additional_growth_rate_pre_2027 = 0  # 0% additional growth until 2027
additional_growth_rate_post_2027 = 0.10  # 10% additional growth after 2027

# Define foreign investment contributions and their lagged effect parameters for each sector
sector_lag_effect_list = [
    ("Горнодобывающая промышленность и разработка карьеров", 4, 1.2),  # starts impacting 4 years after 2027
    ("Обрабатывающая промышленность", 6, 1.5)  # starts impacting 6 years after 2027
]

# Initialize lists to store natural and forecasted GDP values
natural_gdp = [gdp_start]
forecasted_gdp = [gdp_start]

# Calculate natural GDP growth
for year in years[1:]:
    natural_gdp.append(natural_gdp[-1] * (1 + natural_growth_rate))

# Calculate forecasted GDP with foreign investment impact
for i, year in enumerate(years):
    if year < 2027:
        forecasted_gdp.append(forecasted_gdp[-1] * (1 + natural_growth_rate))
    elif year < 2030:
        forecasted_gdp.append(forecasted_gdp[-1] * (1 + natural_growth_rate + additional_growth_rate_pre_2027))
    else:
        # Apply natural growth plus post-2030 additional growth rate
        updated_forecasted_gdp = forecasted_gdp[-1] * (1 + natural_growth_rate + additional_growth_rate_post_2027)

        # Apply foreign investment lagged effects for specified sectors
        for sector, lag_years, boost_factor in sector_lag_effect_list:
            if year >= 2027 + lag_years:
                updated_forecasted_gdp += boost_factor  # Add sector-specific boost

        forecasted_gdp.append(updated_forecasted_gdp)

# Create a DataFrame for plotting
gdp_data = pd.DataFrame({
    "Year": years,
    "Natural GDP (trillion Tenge)": natural_gdp,
    "Forecasted GDP (trillion Tenge)": forecasted_gdp[1:]  # start from 2023 onward
})

doubling_year = gdp_data[gdp_data["Forecasted GDP (trillion Tenge)"] >= gdp_start_doubled]["Year"].iloc[0]


# Plot the GDP trends
plt.figure(figsize=(12, 6))
plt.plot(gdp_data["Year"], gdp_data["Natural GDP (trillion Tenge)"], label="Natural Growth", marker='o')
plt.plot(gdp_data["Year"], gdp_data["Forecasted GDP (trillion Tenge)"], label="Forecasted Growth with Foreign Investment", linestyle="--", marker='x')
plt.axvline(x=2027, color='green', linestyle=':', label="Foreign Investment Effect Starts (2027)")
plt.axvline(x=doubling_year, color='red', linestyle=':', label=f"Year reached: {doubling_year}")

# Add labels and title
plt.title("GDP Growth Trends (2023-2040)", fontsize=14)
plt.xlabel("Year", fontsize=12)
plt.ylabel("GDP (Trillion Tenge)", fontsize=12)
plt.xticks(years, rotation=45)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)

# Show the plot
plt.tight_layout()
plt.savefig('graphs/Predicted_GDP_with_Add_Investments.png')
plt.show()