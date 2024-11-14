import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)

# Loading the data
final_df = pd.read_csv('data/final_df.csv')


#decomposition = seasonal_decompose(final_df['Annual_Percent_Change'], model='additive', period = 4)

# Deseasonalized series (GDP - Seasonal component)
#final_df['GDP_deseasonalized'] = final_df['GDP'] - decomposition.seasonal



# Define target (GDP without the season) and predictors (sector spending columns)
X = final_df.drop(['Annual_Percent_Change', 'GDP', 'Date', 'Quarter', 'Year'], axis = 1)
y = final_df['GDP']

# Include lagged GDP growth
X['GDP_Lag1'] = y.shift(1)
X['GDP_Lag2'] = y.shift(2)
X['GDP_Lag3'] = y.shift(3)

X = X.dropna()
y = y[X.index]

tscv = TimeSeriesSplit(n_splits = 4)

# Define alphas

# LassoCV
lasso = make_pipeline(StandardScaler(),
                      LassoCV(cv = tscv, max_iter = 20000, random_state = 40))


# Fit the model
lasso.fit(X, y)

best_alpha = lasso.named_steps['lassocv'].alpha_
print("Best alpha:", best_alpha)

# Get the coefficients
lasso_coefficients = pd.Series(lasso.named_steps['lassocv'].coef_, index=X.columns)

# Select important features
important_features = lasso_coefficients[lasso_coefficients != 0].sort_values(ascending=False)

#final_df.to_csv('final_df.csv', index = False)
print("The most important features according to Lasso model:")
print(important_features)

ElasticNetCV