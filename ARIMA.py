import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_predict

# Load your dataset
data = pd.read_csv('top7.csv', parse_dates=True, index_col='Date')
close_prices = data['Close']

# Define ARIMA model parameters
p = 1  # lag order (number of time lags to consider for the AR model)
d = 1  # degree of differencing (the number of times the data have had past values subtracted)
q = 1  # order of the moving average

# Fit the ARIMA model
model = ARIMA(close_prices, order=(p, d, q))
fitted_model = model.fit()

# Forecasting
preds = fitted_model.get_forecast(steps=20)
pred_ci = preds.conf_int()

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(close_prices, label='Actual Prices')
plt.plot(preds.predicted_mean, label='Forecast', color='red')
plt.fill_between(pred_ci.index,
                 pred_ci.iloc[:, 0],
                 pred_ci.iloc[:, 1], color='pink', alpha=.3)
plt.title('ARIMA Forecast of Stock Prices')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
