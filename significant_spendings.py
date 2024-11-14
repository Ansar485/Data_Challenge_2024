import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import TimeSeriesSplit
#from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)

# Loading the data
final_df = pd.read_csv('data/final_df.csv')

# Define target (GDP without the season) and predictors (sector spending columns)
X = final_df.drop(['Annual_Percent_Change', 'GDP','Date'], axis = 1)
y = final_df['GDP']

not_sectors = ['Year', 'Quarter', 'GDP', 'Population', 'Inflation', 'Annual_Percentage_Change', 'Date']

investment_columns = [col for col in X.columns if col not in not_sectors]
for col in investment_columns:
    for lag in range(12, 25): # putting lags for the sectors from 3 to 6 years as they have a long-term effect
        X[f'{col}_Lag{lag}'] = X[col].shift(lag)

X = X.drop(investment_columns, axis = 1) # dropping investment columns cuz they don't have the immediate effect

X = X.dropna()
y = y[X.index]

tscv = TimeSeriesSplit(n_splits = 8)

elastic_net = make_pipeline(StandardScaler(),
                            ElasticNetCV(cv=tscv,
                                         max_iter=30000,
                                         random_state=40,
                                         l1_ratio = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1],
                                         ))

# Fit the model
elastic_net.fit(X, y)

# Get the best alpha and l1_ratio
best_alpha = elastic_net.named_steps['elasticnetcv'].alpha_
best_l1_ratio = elastic_net.named_steps['elasticnetcv'].l1_ratio_
print("Best alpha:", best_alpha)
print("Best l1_ratio:", best_l1_ratio)

# Get the coefficients
elastic_net_coefficients = pd.Series(elastic_net.named_steps['elasticnetcv'].coef_, index = X.columns)

# Select important features
important_features = elastic_net_coefficients[elastic_net_coefficients > 0].sort_values(ascending = False)

print("The most important features according to ElasticNet model:")
print(important_features)
