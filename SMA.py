import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset
data = pd.read_csv('top7.csv')
close_prices = data['Close']

# Define the window size for SMA
window_size = 20  # Adjust this based on your analysis needs

# Calculate SMA
sma = close_prices.rolling(window=window_size).mean()

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(close_prices, label='Actual Prices', color='blue')
plt.plot(sma, label='SMA 20', color='red', linestyle='--')
plt.title('Simple Moving Average vs Actual Prices')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
