import yfinance as yf

# Define the ticker symbol
ticker_symbol = 'AAPL'  # Apple Inc.

# Get historical data for this ticker
ticker_data = yf.Ticker(ticker_symbol)
ticker_df = ticker_data.history(period='1y')  # Get the last year of stock data

# Save the data to a CSV file
ticker_df.to_csv('top7.csv')

print("Data downloaded and saved to AAPL_stock_data.csv")
