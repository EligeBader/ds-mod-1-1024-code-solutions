# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller


# %%
data = pd.read_csv("AirPassengers.csv")
data.head()

# %%
data.info()

# %%
data['Month'] = pd.to_datetime(data['Month'])
data.info()

# %%
data.head()

# %%
results = adfuller(data['Month']) 
results #this is nit a stationary data pvalue > 0.05

# %%
data.set_index('Month', inplace=True) 

# %%
data

# %%
data.plot() # This data is not stationary

# %% [markdown]
# Plot the time series data with rolling mean and rolling standard deviation and see if it is stationary.

# %%
data['rolling_mean'] = data['#Passengers'].rolling(7).mean() 
data['rolling_std'] = data['#Passengers'].rolling(7).std() 

data.plot(); # change in variance, trend of the mean, change in auto-covariance

# %% [markdown]
# Observation: This data is not stationary having a trend in mean

# %% [markdown]
# Try different levels of differences, and plot the time series data with rolling mean and standard deviation. See if it is stationary.

# %%
data['New_#Passengers'] = data['#Passengers'].shift()
data

# %%
data['diff1'] = data['#Passengers'].diff()
data

# %%
data['diff2'] = data['#Passengers'].diff(2)
data

# %%
data['diff3'] = data['#Passengers'].diff(3)
data

# %%
data['diff1'].plot(); # no trend but change in variance so it is not stationary

# %%
data['diff2'].plot(); # no trend but change in variance so it is not stationary

# %%
data['diff3'].plot(); # no trend but change in variance so it is not stationary

# %% [markdown]
# Try to transform the data, and make different levels of differences. See if it is stationary.

# %%
data['#Passengers'].plot()

# %%
data['log_#Passengers'] = np.log(data['#Passengers'])

# %%
data['log_#Passengers'].plot();

# %%
data['NewLog_#Passengers'] = data['log_#Passengers'].shift()
data

# %%
data['diff1_log'] = data['log_#Passengers'].diff()
data['diff1_log'].plot()

# %%
data['diff2_log'] = data['log_#Passengers'].diff(2)
data['diff2_log'].plot();

# %%
data['diff3_log'] = data['log_#Passengers'].diff(3)
data['diff3_log'].plot();

# %%
adf = adfuller(data['log_#Passengers'].dropna())
adf # not stationary 

# %%
adf1 = adfuller(data['diff1_log'].dropna())
adf1 #not stationary

# %%
adf2 = adfuller(data['diff2_log'].dropna())
adf2  # stationary

# %%
adf3 = adfuller(data['diff3_log'].dropna())
adf3 #not stationary


