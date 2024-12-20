# %% [markdown]
# # Libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import root_mean_squared_error
import itertools 
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime
import pytz


# %% [markdown]
# # Read data

# %%
stock =  yf.download('AAPL', period='1Y', interval='1h')
stock

# %%
type(stock.index)

# %%
stock.index.hour


# %%
# Convert the Datetime column to EST
stock.index =  stock.index.tz_convert('US/Eastern')

# %%
stock.index

# %%
# Filter rows between 8 AM and 4 PM EST 
start_time = pd.Timestamp('08:00').time() 
end_time = pd.Timestamp('16:00').time()

# %%
# Function to filter the rows 
def is_within_business_hours(timestamp): 
    return start_time <= timestamp.time() <= end_time

# %%
filtered_stock = stock[stock.index.map(is_within_business_hours)]

# %%
filtered_stock

# %%
close = filtered_stock['Close']
close

# %%
close.isna().sum()

# %% [markdown]
# # Analyse Data

# %% [markdown]
# ### Check if stationary

# %%
close.plot(); # not stationary having trend & change in variance

# %%
adfuller(close)

# %%
close_log = np.log(close)
close_log.plot();

# %%
adfuller(close_log)

# %%
close_log_diff1 = close_log.diff()
close_log_diff1.plot();

# %%
adfuller(close_log_diff1.dropna()) # stationary

# %% [markdown]
# ### Check seasonality

# %%
plot_acf(close_log_diff1.dropna(), lags=np.arange(0,100));
plt.ylim(-0.2, 0.2)

# %%
plot_pacf(close_log_diff1.dropna());
plt.ylim(-0.2, 0.2)

# %%
close = close.asfreq('H')

# %%
close.isna().sum()

# %%
ffill = close.fillna(method='ffill')
bfill = close.fillna(method='bfill')
close_filled = (ffill + bfill)/2
close_filled.head(25)


# %%
# use original data to get seasoning:
decomp = seasonal_decompose(close_filled)

# %%
trend =  decomp.trend
seasonal = decomp.seasonal
noise = decomp.resid

# %%
fig, ax = plt.subplots(nrows=4)

ax[0].plot(close_filled);
ax[1].plot(trend);
ax[2].plot(seasonal);
ax[3].plot(noise);

plt.tight_layout()

# %%
noise.plot();

# %%
adfuller(noise.dropna()) # stationary

# %%
noise_log_diff2 = np.log(noise).diff(2)
noise_log_diff2.plot();

# %%
adfuller(noise_log_diff2.dropna())

# %%
# Find P, D, Q
plot_acf(noise.dropna(), lags=np.arange(0,50)); 

# %%
q_vals = [1, 2, 3]

# %%
plot_pacf(noise.dropna(), lags=np.arange(0,30));
plt.ylim(-.4, .4);


# %%
p_vals = [1,2]

# %%
seasonal[0:200].plot();

# %%
adfuller(seasonal.dropna())

# %%
# plt.subplots(figsize = (15,5))
# plt.plot(seasonal[0:14].index, seasonal[0:14]);

# %%
plot_acf(seasonal.dropna(), lags = np.arange(0,50));

# %%
sq_vals = 2

# %%
plot_pacf(seasonal.dropna(), lags=np.arange(0,25));


# %%
plot_acf(close_log_diff1.dropna(), lags=np.arange(0,100));
plt.ylim(-0.2, 0.2);

# %%
plot_pacf(close_log_diff1.dropna(), lags=np.arange(0,100));
plt.ylim(-0.2, 0.2)

# %%
p_vals = [1, 2, 3]
d_vals = [1, 2]
q_vals = [0, 1, 2] 

# %%
best_rmse = np.inf 
best_pdq = (-1, -1, -1)

train = close_log[:len(close_log)-50]
test = close_log[len(close_log)-50:] 

for pdq in itertools.product(p_vals, d_vals, q_vals):
    model =  ARIMA(train, order=pdq).fit()

    fv = model.fittedvalues
    fv_exp = np.exp(fv)

    test_inv = np.exp(test)
    model_inv = np.exp(model.forecast(50))

    rmse = root_mean_squared_error(test_inv, model_inv)
    
    if rmse < best_rmse:
        best_rmse = rmse
        best_pdq = pdq
        best_model = model

# %%
best_rmse

# %%
best_pdq

# %%
# Predict the next two timestamps 
forecast = best_model.get_forecast(steps=50) 
forecast_mean = forecast.predicted_mean 
forecast_exp = np.exp(forecast_mean) 
print("Predicted prices for the next two timestamps:") 
print(forecast_exp)

# %%
np.exp(close_log[2:100]).plot() # plot the whole dataset
np.exp(best_model.fittedvalues[2:100]).plot()

# %%
plt.plot(test.index, np.exp(test.values))
plt.plot(test.index, np.exp(best_model.forecast(50)))

# %%
close.plot()
plt.plot(train.index, np.exp(best_model.fittedvalues.values))
plt.plot(test.index, np.exp(best_model.forecast(50)))

# %%
# Slice the last 100 rows
close_sliced = close[100:] 

train_sliced = train[100:] 
test_50 = test[:50] 

# Using the first 50 points of the test set - Plot the sliced data 
plt.figure(figsize=(12, 6)) 
plt.plot(close_sliced.index, close_sliced, label='Actual Close Prices') 

# Plot the training data fitted values for the last 100 rows
plt.plot(train_sliced.index, np.exp(best_model.fittedvalues[100:]), label='Fitted Values (Training Data)') 

# Plot the forecasted values for the test period 
forecasted_values = best_model.get_forecast(steps=50).predicted_mean 
plt.plot(test_50 .index, np.exp(forecasted_values), label='Forecasted Values') 


plt.title('Stock Price Prediction') 
plt.xlabel('Date') 
plt.ylabel('Price') 
plt.legend() 
plt.grid(True)


