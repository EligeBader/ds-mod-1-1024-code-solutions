# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

# %%
data = pd.read_csv("AirPassengers.csv")
data

# %%
data['Month'] = pd.to_datetime(data['Month'])
data = data.set_index('Month')
data

# %%
data.plot();

# %%
data_log = np.log(data)
data_log.plot();

# %%
data_log_diff1 = data_log.diff()
data_log_diff1.plot();

# %%
adfuller(data_log_diff1.dropna()) # not stationary


# %%
data_log_diff2 = data_log_diff1.diff(2)
data_log_diff2.plot();

# %%
adfuller(data_log_diff2.dropna()) # stationary

# %%
plot_acf(data_log_diff2.dropna(), lags= np.arange(0, 50));  #i can look into highest values every 12 month

# %%
plot_pacf(data_log_diff2.dropna(), lags= np.arange(0, 50)); # look at ar10 model


# %%
import itertools 
from sklearn.metrics import root_mean_squared_error

# %%
p_vals = [0, 10] 
d_vals = [1]
q_vals = [0, 12, 24, 36, 48] 

# %%
best_rmse = np.inf #initialize it to be the worst rmse
best_pdq = (0, 0, 0)
for pdq in itertools.product(p_vals, d_vals, q_vals):
    model =  ARIMA(data_log, order=pdq).fit()
    fv = model.fittedvalues
    fv_exp = np.exp(fv)
    rmse = root_mean_squared_error(data, fv_exp)
    if rmse < best_rmse:
        best_rmse = rmse
        best_pdq = pdq

# %%
best_rmse

# %%
best_pdq

# %%
np.exp(model.forecast(20))

# %%
data.plot()
fv_exp.plot()

# %% [markdown]
# ## Seasonality

# %%
from statsmodels.tsa.seasonal import seasonal_decompose

# %%
decomp = seasonal_decompose(data)

# %%
trend = decomp.trend
seasonal = decomp.seasonal
noise = decomp.resid

# %%
fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(10, 6)) 
ax[0].plot(data.index, data.values)
ax[1].plot(data.index, trend)
ax[2].plot(data.index, seasonal)
ax[3].plot(data.index, noise)

plt.tight_layout()

# %%
adfuller(noise.dropna()) 

# %%
noise.plot(); # doesn't look stationary

# %%
noise_diff1 = noise.diff()
noise_diff1.plot();

# %%
adfuller(noise_diff1.dropna())

# %%
plot_acf(noise_diff1.dropna()); #q =12

# %%
plot_pacf(noise_diff1.dropna()); # p =12

# %%
ar8 = ARIMA(noise, order = (8, 1, 0)).fit()
am8 = ARIMA(noise, order = (0, 1, 8)).fit()
ar8am8 = ARIMA(noise, order = (8, 1, 8)).fit()

# %%
noise.plot()
ar8.fittedvalues.plot()

# %%
noise.plot()
am8.fittedvalues.plot()

# %%
noise.plot()
ar8am8.fittedvalues.plot()

# %%
seasonal.plot();

# %%
from statsmodels.tsa.statespace.sarimax import SARIMAX 

# %%
adfuller(seasonal)

# %%
plot_acf(seasonal.dropna(), lags=np.arange(0,50)); 

# %%
plot_pacf(seasonal.dropna(), lags=np.arange(0,50));

# %%
model = SARIMAX(data, order = (8, 1, 8), seasonal_order =(1, 0, 0, 12), trend = 'ct').fit() 
# use for order (8, 1 ,8) or (11, 1, 11) as we can't use (12, 1, 12) when period s=12 

# %%
model.fittedvalues

# %%
data.plot()
model.fittedvalues.plot();

# %%
rmse = root_mean_squared_error(data, model.fittedvalues)
rmse


