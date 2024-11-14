import pandas as pd
import numpy as np

# Spendings Data
df_spendings = pd.read_excel('data/инвестиции помесячно.xls', header = 10)
columns_to_clean = df_spendings.columns[1:]

df_spendings[columns_to_clean] = df_spendings[columns_to_clean].apply(lambda x: x.str.replace('.', '').str.replace('x', ''))
df_spendings[columns_to_clean] = df_spendings[columns_to_clean].replace(r'^\s*$', np.nan, regex=True)
df_spendings[columns_to_clean] = df_spendings[columns_to_clean].astype(float)/1000 # to convert into millions
df_spendings.rename(columns = {'Регион': 'Sector'}, inplace = True)
df_spendings = df_spendings[~df_spendings['Sector'].str.contains('Промышленность|Всего', regex=True)]
df_spendings['Sector'] = df_spendings['Sector'].str.strip() # deleting all the whitespace on the left and right
df_spendings['Sector'] = df_spendings['Sector'].str.replace('  ', ' ') # deleting excessive spaces

df_spendings = df_spendings.melt(id_vars=['Sector'], var_name='Year_Month', value_name='Value')
df_spendings['Year'] = df_spendings['Year_Month'].str.extract(r'(\d{4})').astype(int)
df_spendings['Month'] = df_spendings['Year_Month'].str.extract(r'([А-Яа-я]+)')
df_spendings['Quarter'] = df_spendings['Month'].map({
    'Январь': 1, 'Февраль': 1, 'Март': 1,
    'Апрель': 2, 'Май': 2, 'Июнь': 2,
    'Июль': 3, 'Август': 3, 'Сентябрь': 3,
    'Октябрь': 4, 'Ноябрь': 4, 'Декабрь': 4
})

# Group by Year, Quarter, and Sector, then sum the values
df_spendings = df_spendings.groupby(['Year', 'Quarter', 'Sector'])['Value'].sum().reset_index()

# Pivot the table to get sectors as columns
df_spendings = df_spendings.pivot(index=['Year', 'Quarter'], columns='Sector', values='Value').reset_index()

# GDP Data
df_gdp = pd.read_excel('data/Данные 2024.11.10-15.36.40..xls', header = 10) # total gdp is in millions, nominal
df_gdp = df_gdp[df_gdp['Регион'] == 'РЕСПУБЛИКА КАЗАХСТАН']
df_gdp = df_gdp.melt(
    value_vars = df_gdp.columns[1:],
    value_name = 'GDP',
    var_name = 'Year_Period'
)

df_gdp['GDP'] = df_gdp['GDP'].str.replace('.', '').str.replace(',', '.').astype(float)
df_gdp['Year_Period'] = df_gdp['Year_Period'].str.strip()
df_gdp['Year'] = df_gdp['Year_Period'].str.extract(r'(\d{4})').astype(int)
df_gdp['Period'] = np.where(df_gdp['Year_Period'].str.contains('Январь - Март'), 1,
                    np.where(df_gdp['Year_Period'].str.contains('Апрель - Июнь'), 2,
                    np.where(df_gdp['Year_Period'].str.contains('Июль - Сентябрь'), 3, 4)))

df_gdp['GDP_Non_Cumulative'] = df_gdp.groupby('Year')['GDP'].diff()

# For the first period of each year, use the original GDP value
df_gdp.loc[df_gdp['Period'] == 1, 'GDP_Non_Cumulative'] = df_gdp.loc[df_gdp['Period'] == 1, 'GDP']
df_gdp['GDP'] = df_gdp['GDP_Non_Cumulative']
df_gdp = df_gdp.drop('GDP_Non_Cumulative', axis = 1)

df_gdp['Quarter'] = np.where(df_gdp['Year_Period'].str.contains('Март'), 1,
                    np.where(df_gdp['Year_Period'].str.contains('Июнь'), 2,
                    np.where(df_gdp['Year_Period'].str.contains('Сентябрь'), 3, 4)))

df_gdp = df_gdp[['Year', 'Quarter', 'GDP']]


# Population Data
df_population = pd.read_excel('data/Численность_населения_на_начало_периода.xls', header = 10).iloc[:, 1:]

df_population = df_population.melt(
    value_vars = df_population.columns,
    value_name = 'Population',
    var_name = 'Year_Month'
).reset_index(drop = True)

df_population['Quarter'] = np.where(df_population['Year_Month'].str.contains('Март'), 1,
                    np.where(df_population['Year_Month'].str.contains('Июнь'), 2,
                    np.where(df_population['Year_Month'].str.contains('Сентябрь'), 3,
                    np.where(df_population['Year_Month'].str.contains('Декабрь'), 4, np.nan))))

df_population = df_population.dropna().reset_index(drop = True)
df_population['Year'] = df_population['Year_Month'].str.extract(r'(\d{4})').astype(int)
df_population['Population'] = df_population['Population'].str.replace('.', '').astype(int)
df_population = df_population[['Year', 'Quarter', 'Population']]

# Inflation Data
df_infl = pd.read_excel('data/Инфляция.xlsx')

def expand_to_quarterly(yearly_data):
    # Create a list to store the quarterly data
    quarterly_data = []

    # Iterate through each year
    for _, row in yearly_data.iterrows():
        year = row['Year']
        inflation = row['Inflation']

        # Add an entry for each quarter
        for quarter in range(1, 5):
            quarterly_data.append({
                'Year': year,
                'Quarter': quarter,
                'Inflation': inflation
            })

    # Convert to DataFrame and sort by Year and Quarter
    quarterly_df = pd.DataFrame(quarterly_data)
    quarterly_df = quarterly_df.sort_values(['Year', 'Quarter']).reset_index(drop=True)

    return quarterly_df

df_infl = expand_to_quarterly(df_infl)

# Now merge with df_gdp
final_df = df_spendings.merge(
    df_gdp,
    on = ['Year', 'Quarter'],
    how = 'inner'  # or 'inner' if you only want years that exist in both
).merge(
    df_population,
    on = ['Year', 'Quarter'],
    how = 'inner'
).merge(
    df_infl,
    on = ['Year', 'Quarter'],
    how = 'inner'
)

sector_translation = {
    'Сельское, лесное и рыбное хозяйство': 'Agriculture_Forestry_Fishing',
    'Промышленность': 'Industry',
    'Строительство': 'Construction',
    'Оптовая и розничнаяторговля; ремонт автомобилей и мотоциклов': 'Wholesale_Retail_Trade',
    'Транспорт и складирование': 'Transport_Storage',
    'Предоставление услуг по проживанию и питанию': 'Accommodation_Food_Services',
    'Информация и связь': 'Info_Communication',
    'Финансовая и страховая деятельность': 'Finance_Insurance',
    'Операции с недвижимым имуществом': 'Real_Estate',
    'Профессиональная, научная и техническая деятельность': 'Professional_Technical_Services',
    'Деятельность в области административного и вспомогательного обслуживания': 'Admin_Support_Services',
    'Государственное управление и оборона; обязательное социальное обеспечение': 'Public_Admin_Defense',
    'Образование': 'Education',
    'Здравоохранение и социальное обслуживание населения': 'Healthcare_Social_Services',
    'Искусство, развлечения и отдых': 'Arts_Entertainment_Recreation',
    'Предоставление прочих видов услуг': 'Other_Services',
    'Деятельность домашних хозяйств, нанимающих домашнюю прислугу; деятельность домашних хозяйств по производству товаров и услуг для собственного потребления': 'Household_Services',
    'Итого по отраслям': 'Total_Industries',
    'Косвенно-измеряемые услуги финансового посредничества': 'Imputed_Financial_Services',
    'Валовая добавленная стоимость': 'Gross_Value_Added',
    'Чистые налоги на продукты и импорт': 'Net_Taxes_Products_Import',
    'Налоги на продукты и импорт': 'Taxes_Products_Import',
    'Субсидии на продукты и импорт': 'Subsidies_Products_Import',
    'Валовой внутренний продукт': 'Gross_Domestic_Product'
}

final_df.rename(columns = sector_translation, inplace = True)
final_df['Annual_Percent_Change'] = final_df['GDP'].pct_change(4) * 100

# Drop rows with NaN values in 'GDP_Growth' (the first year)
final_df = final_df.dropna(subset=['Annual_Percent_Change']).reset_index(drop = True)

final_df['Date'] = pd.to_datetime(final_df['Year'].astype(str) + '-Q' + final_df['Quarter'].astype(str)) + pd.offsets.QuarterBegin(startingMonth = 1)
#final_df['Annual_GDP_Diff'] = final_df['GDP'] - final_df['GDP'].shift(4)

final_df.to_csv('data/final_df.csv', index = False)