import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

#demo dataset
df = pd.read_csv("advertising.csv")

#independent variables  and dependent variable 
X = df[['TV', 'Radio', 'Newspaper']]  #Predictor variables
y = df['Sales']  #Target variable
X = sm.add_constant(X)
# Fit the Multiple Linear Regression Model
model = sm.OLS(y, X).fit()
r_squared = model.rsquared
rss = sum((model.resid) ** 2)  # Residual Sum of Squares
rse = np.sqrt(rss / (len(y) - len(X.columns)))  # RSE formula

#Get F-statistics
f_statistic = model.fvalue
p_value = model.f_pvalue

#results
print(f"R² Value: {r_squared:.4f}")
print(f"Residual Standard Error (RSE): {rse:.4f}")
print(f"F-Statistic: {f_statistic:.4f}")
print(f"P-Value (F-Test): {p_value:.4e}")

#Model summary
print(model.summary())
