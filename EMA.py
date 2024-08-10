import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset
data = pd.read_csv('top7.csv')
close_prices = data['Close']

# Define the span for EMA, which corresponds to the "half-life" or the decay factor
span = 20  # This can be adjusted based on your analysis needs

# Calculate EMA
ema = close_prices.ewm(span=span, adjust=False).mean()

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(close_prices, label='Actual Prices', color='blue')
plt.plot(ema, label='EMA 20', color='green', linestyle='--')
plt.title('Exponential Moving Average vs Actual Prices')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
